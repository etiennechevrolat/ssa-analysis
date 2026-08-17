### SETUP 
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import hydra 
import math 
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig


from ml.datahandler import load_splid_objects, load_spacetrack_objects, load_doris_objects
from ml.dataset import make_loaders, make_loaders_classifiers, make_pretrain_loader, make_loaders_finetuning
from ml.model import build_model
from ml.targets import doris_types, intensity_labels

from ml.utils import to_device, compute_class_weights, MaskedChannelMSE

from ml.logger import RunLogger


def train_one_epoch(
        model : nn.Module,
        data_loader : DataLoader,
        loss_fn : nn.Module,
        optimizer : torch.optim.Optimizer,
        device : torch.device,
        epoch=0,
        scheduler=None
    ):
    model.train()
    running_loss = 0.0
    total = 0

    bar = tqdm(data_loader, desc=f"train {epoch}", leave=False)

    for time_series_batch, labels in bar:
        # Localizer :
        ## time_series_batch: (B,F,W), labels (B,2) ou (B,4) pour DORIS finetuning

        # Classifier :
        ## time_series_batch (B,F,W), labels (B, dict = {'node' : , 'class':  })
        x = time_series_batch.to(device, non_blocking=True)
        y = to_device(labels, device)

        optimizer.zero_grad()

        pred = model(x)


        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * x.size(0)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += x.size(0)
        bar.set_postfix(loss=f"{running_loss / total :.4f}")

    avg_loss = running_loss / total
 
    return avg_loss 


def pretrain_one_epoch(
        model : nn.Module,
        data_loader : DataLoader,
        loss_fn : nn.Module, 
        optimizer : torch.optim.Optimizer,
        device : torch.device,
        epoch=0,
        scheduler=None
    ):
    ## dataloader renvoie des batchs de times series sans label de type UnlabelledWindowDataset() cf dataset.py
    model.train()
    running_loss = 0.0
    total = 0

    bar = tqdm(data_loader, desc=f"train {epoch}", leave=False)

    for time_series_batch in bar:
       
        x = time_series_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred, target = model(x) 

        ## on évalue donc l'image initiale vs la reconstruction 
        loss = loss_fn(pred, target)
        running_loss+= float(loss.item()) * x.size(0) 
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        total += x.size(0)
        bar.set_postfix(loss=f"{running_loss / total :.4f}")

    avg_loss = running_loss / total
 
    return avg_loss 

from ml.evaluate import matching_tolerance, extract_events, gt_events_from_labels, evaluate_predictions

@torch.no_grad() ## décorateur, applique torch.no_grad(evaluate_epoch(...)) et coupe le suivi des gradients.
def evaluate_epoch(
    model,
    data_loader,
    meta,
    labels,
    loss_fn, 
    device
    ):
    model.eval() ## différents de torch.no_grad, change le comportement de certaines couches : Dropout, BatchNorm.
    running_loss = 0.0
    total=0
    all_probs = []
    for time_series_batch, labels_batch in tqdm(data_loader, desc="val", leave=False):
        x = time_series_batch.to(device)
        y = labels_batch.to(device)

        pred = model(x) ## format (Batch_size, 2) ou (B,4) pour DORIS 

        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels_batch.size(0)
        total += labels_batch.size(0)
        
        all_probs.append(torch.sigmoid(pred).cpu().numpy())

    
    ## On met les probas au format attendu pour extract_events (L,2) avec L longueur d'UN objet : découpage de la data par objet
    all_probs = np.concatenate(all_probs, axis=0) ## (N,2)
    val_ids = meta['val_ids']
    objects_lengths = [len(meta['per_obj'][oid][0]) for oid in val_ids]

    assert(np.sum(objects_lengths) == len(all_probs)) ## on s'assure de la correspondance des tailles.

    seq_per_obj = np.split(all_probs, np.cumsum(objects_lengths)[:-1], axis=0) ## listes (L1,2) (L2, 2) ... (Li, 2) de probas par objet

    

    pred_events = {oid : extract_events(seq) for oid, seq in zip(val_ids, seq_per_obj)}
    gt_events = gt_events_from_labels(labels, val_ids) ## labels est ici le dataframe brut rendu par load_splid_objects

    metrics = evaluate_predictions(gt_events, pred_events) ## renvoie dictionnaire des métriques de performance
    
    avg_loss = running_loss / total
    return avg_loss, metrics



@torch.no_grad()
def evaluate_epoch_finetuning(
    model,
    data_loader,
    loss_fn,
    device
    ):
    """
    Validation du finetuning DORIS.
    Métrique PONCTUELLE (par pas de temps), et non le protocole d'appariement d'évènements
    utilisé pour le localizer SPLID : extract_events itère sur des colonnes EW/NS et
    gt_events_from_labels attend un dataframe SPLID (ObjectID/Direction/Node), ni l'un ni
    l'autre ne s'applique à une cible (L, 2, 2) et à des labels norad_id/maneuver_type.
    On utilise l'average precision, sans seuil, pour ne pas figer un seuil de détection
    avant d'avoir regardé les courbes.
    """
    model.eval()
    running_loss = 0.0
    total = 0
    all_scores, all_targets = [], []

    for time_series_batch, labels_batch in tqdm(data_loader, desc="val", leave=False):
        x = time_series_batch.to(device)
        y = labels_batch.to(device)

        pred = model(x) ## (B, 2*C) logits
        loss = loss_fn(pred, y)
        running_loss += float(loss.item()) * labels_batch.size(0)
        total += labels_batch.size(0)

        all_scores.append(torch.sigmoid(pred).cpu().numpy())
        all_targets.append(labels_batch.numpy())

    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    ## les cibles sont des bosses triangulaires continues : est positif tout point sous une bosse
    positives = targets > 0
    channel_names = [f"{maneuver_type}/{intensity}"
                     for maneuver_type in doris_types for intensity in intensity_labels]

    metrics, aps = {}, []
    for col, name in enumerate(channel_names[:scores.shape[1]]):
        if positives[:, col].any():
            ap = float(average_precision_score(positives[:, col], scores[:, col]))
            aps.append(ap)
        else:
            ap = float('nan') ## canal absent de la validation : aucune manoeuvre de ce type/intensité
        metrics[f"ap {name}"] = ap
    metrics["mAP"] = float(np.mean(aps)) if aps else float('nan')

    return running_loss / total, metrics


def check_backbone_compatibility(ckpt, cfg, window_size):
    """
    Vérifie que le backbone pré-entraîné a la même géométrie que le modèle de finetuning.
    Sans ça, load_state_dict échoue sur un mur de shape mismatch dont la cause réelle
    (taille de fenêtre, patch_size, embed_dim) n'apparaît nulle part.
    """
    pretrain_cfg = ckpt.get('config')
    if pretrain_cfg is None:
        print("[finetuning] checkpoint sans config : compatibilité du backbone non vérifiée")
        return
    try:
        compared = {
            'window_size'  : (pretrain_cfg['data']['window_size'], window_size),
            'patch_size'   : (pretrain_cfg['model']['patch_size'], cfg.model.patch_size),
            'embed_dim'    : (pretrain_cfg['model']['encoder_embed_dim'], cfg.model.embed_dim),
            'n_blocks'     : (pretrain_cfg['model']['encoder_n_blocks'], cfg.model.n_blocks),
            'n_attn_heads' : (pretrain_cfg['model']['encoder_n_attn_heads'], cfg.model.n_attn_heads),
            }
    except KeyError as e:
        print(f"[finetuning] clé {e} absente de la config du checkpoint : vérification partielle impossible")
        return

    mismatch = {key : values for key, values in compared.items() if values[0] != values[1]}
    if mismatch:
        detail = ", ".join(f"{key} : pretrain={a} vs finetuning={b}" for key, (a, b) in mismatch.items())
        raise ValueError(f"backbone incompatible avec le modèle de finetuning ({detail})")


@torch.no_grad()
def evaluate_epoch_classifier(
    model,
    data_loader,
    n_node_types,
    n_classes,
    loss_fn, 
    device
    ):

    model.eval()
    running_loss, total = 0.0, 0
    heads = ('node', 'class')

    n_labels = {'node' : n_node_types, 'class' : n_classes}
    y_true = {h : [] for h in heads}
    y_pred = {h : [] for h in heads}

    for time_series_batch, node_samples_batch in tqdm(data_loader, desc="val", leave=False):
        x = time_series_batch.to(device)
        y = to_device(node_samples_batch, device) 

        logits = model(x) ## {'node' : (B,n_node), 'class' : (B,n_class)}
        loss = loss_fn(logits, y)
        batch_size = x.size(0)

        running_loss += float(loss.item()) * batch_size
        total += batch_size

        for h in heads : 
            y_true[h].append(node_samples_batch[h].cpu().numpy())

            prediction = logits[h].argmax(dim=1).cpu().numpy()
            y_pred[h].append(prediction)
    avg_loss = running_loss / total
    metrics = {}

    ## on utilise directement les métriques scikit_learn pour de la classification classique 
    for h in heads:
        y_t = np.concatenate(y_true[h])
        y_p = np.concatenate(y_pred[h])
        metrics[f"{h}_acc"] = accuracy_score(y_t, y_p)
        metrics[f"{h}_f1"] = f1_score(y_t, y_p, labels=range(n_labels[h]), average='macro', zero_division=0)

    return avg_loss, metrics

@torch.no_grad()
def evaluate_pretraining_epoch(
    model,
    data_loader,
    loss_fn, 
    device,
    seed = 0
    ):
    model.eval() ## différents de torch.no_grad, change le comportement de certaines couches : Dropout, BatchNorm.
    running_loss = 0.0
    total=0

    # on fige le rng le temps de l'évaluation, sinon le masque tiré aléatoirement à chaque forward fait varier la val loss 
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    try : 
        for time_series_batch in tqdm(data_loader, desc="val", leave=False):
            x = time_series_batch.to(device, non_blocking=True)
            pred, target = model(x)
            loss = loss_fn(pred, target)
            running_loss+= float(loss.item()) * x.size(0)
            total += x.size(0)
    finally :
        torch.set_rng_state(rng_state)

    avg_loss = running_loss / total
    return avg_loss




@hydra.main(version_base=None, config_path="../../configs/ml", config_name="config") 
## hydra prend main en point d'entrée, et main() est transformé en wrapper qui récupère et construit l'objet cfg, et enfin applique main(cfg).
def main(cfg : DictConfig):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    task = cfg.task.name
    is_classifier = (task == 'classifier')
    is_pretrain = (task == 'pretrain')
    is_finetuning = (task == 'finetuning')


    ## Recup des données et creation des dataloaders
    if is_pretrain :
        objects = load_spacetrack_objects(cfg.data.data_dir)
    elif is_finetuning : 
        objects, labels = load_doris_objects(cfg.data.data_dir)
    else:
        objects, labels = load_splid_objects(
            cfg.data.data_dir, 
            cfg.data.labels_dir,
            )

    if is_classifier :
        train_loader, val_loader, meta = make_loaders_classifiers(
            objects,
            labels,
            batch_size=cfg.train.batch_size,
            history=cfg.data.history,
            future=cfg.data.future,
            val_split=cfg.data.val_split,
            seed= cfg.seed)
        ## taille de la fenetre temporelle pour le dataset splid (différente de la fenetre pretrain spacetrack)
        window_size = cfg.data.history + cfg.data.future + 1

        ## y est un dict contenant 'node' : n_node, 'class' : n_class
        w_class = compute_class_weights(train_loader.dataset, field = 4, n_classes = cfg.task.node_classes, device=device)

        node_loss = nn.CrossEntropyLoss()
        class_loss = nn.CrossEntropyLoss(weight=w_class)
        def loss_fn(pred, y):
            return node_loss(pred['node'], y['node']) + class_loss(pred['class'], y['class'])

    elif is_pretrain :
        window_size = cfg.data.window_size
        train_loader, val_loader, meta = make_pretrain_loader(
            objects,
            window_size=cfg.data.window_size,
            stride = cfg.data.stride,
            batch_size=cfg.train.batch_size,
            val_split=cfg.data.val_split,
            seed= cfg.seed
            )
        ## le pretrain est une régression : mean-square-error loss
        w = torch.ones(len(meta["feature_cols"]))
        w[meta['feature_cols'].index("dt")] = 0.0 ## on mets le poids de dt à zero.
        loss_fn = MaskedChannelMSE(w).to(device)

    elif is_finetuning : 
        train_loader, val_loader, meta = make_loaders_finetuning(
                    objects,
                    labels,
                    batch_size=cfg.train.batch_size,
                    history=cfg.data.history,
                    future=cfg.data.future,
                    val_split=cfg.data.val_split,
                    seed=cfg.seed,
                    half_width=cfg.task.half_width, 
                    flatten_target=True, 
        )
        window_size = cfg.data.history + cfg.data.future + 1
        loss_fn = nn.BCEWithLogitsLoss() 
    else:
        train_loader, val_loader, meta = make_loaders(
                    objects,
                    labels,
                    batch_size=cfg.train.batch_size,
                    history=cfg.data.history,
                    future=cfg.data.future,
                    val_split=cfg.data.val_split,
                    seed= cfg.seed,
                    half_width=cfg.task.half_width)
        window_size  = cfg.data.history + cfg.data.future +1
        loss_fn = nn.BCEWithLogitsLoss()

    ## On calcule le nombre de features considérées
    n_features = len(meta["feature_cols"])


    model = build_model(cfg.model, cfg.task, n_features=n_features, window_size=window_size)
    model.to(device)
    if is_finetuning :
        if not cfg.task.ckpt_path:
            raise ValueError(
                "finetuning sans task.ckpt_path : l'encodeur resterait aléatoire. "
                "Passe le chemin d'un checkpoint de pretrain (outputs/ml/pretrain/<date>/checkpoints/best.pt)"
                )

        ckpt = torch.load(cfg.task.ckpt_path, map_location='cpu', weights_only=False)
        check_backbone_compatibility(ckpt, cfg, window_size)
        model.encoder.load_state_dict(ckpt['encoder_state'], strict=False)

        if cfg.task.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[finetuning] backbone {cfg.task.ckpt_path} chargé | "
              f"encodeur {'gelé' if cfg.task.freeze_encoder else 'entraîné'} | "
              f"paramètres entraînables : {n_trainable} / {n_total}")


    def build_param_groups(model, weight_decay):
        no_decay_exact={"cls_token", "mask_token"}
        decay, no_decay = [], []
        for name,p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name in no_decay_exact or "pos_embedding" in name:
                no_decay.append(p)
            else : 
                decay.append(p)
        return [
            {"params": decay, "weight_decay":weight_decay}, {"params":no_decay, "weight_decay":0.0}
        ]
    ## OPTIMIZER
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.train.weight_decay),
        lr = cfg.train.lr, 
        betas=(0.9, 0.95)
        )
    
    def build_lr_scheduler(optimizer, total_steps, warmup_ratio=0.05, min_lr_ratio=0.0):
        """
        Warmup linéaire puis décroissance cosine, mis à jour à chaque pas d'optimisation
        """
        warmup_steps = max(1, int(total_steps*warmup_ratio))

        def lr_lambda(step):
            if step < warmup_steps :
                return (step  + 1)/ warmup_steps
            progress = (step - warmup_steps) / (max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    ## SCHEDULER 
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = build_lr_scheduler(optimizer, total_steps, warmup_ratio=cfg.train.warmup_epochs / cfg.train.epochs)

    logger = RunLogger(cfg, HydraConfig.get().runtime.output_dir, )
    logger.watch(model)

    for epoch in tqdm(range(cfg.train.epochs), desc="epochs"):

        if is_pretrain : 
            train_loss = pretrain_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch=epoch, scheduler=scheduler)
            val_loss = evaluate_pretraining_epoch(model=model, data_loader=val_loader, loss_fn=loss_fn, device=device, seed=cfg.seed ) 
            metrics = {"lr": scheduler.get_last_lr()[0]}
            line = (f"epoch {epoch} : train loss: {train_loss:.4f} | val loss : {val_loss:.4f} | ")
        else : 
            train_loss= train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch=epoch, scheduler=scheduler)

            if is_classifier :
                val_loss, metrics = evaluate_epoch_classifier(model, val_loader, cfg.task.node_types, cfg.task.node_classes, loss_fn, device)

                line = (f"epoch {epoch} : train loss: {train_loss:.4f} | val loss : {val_loss:.4f} | " 
                    f"node type acc : {metrics['node_acc']:.3f}, f1 : {metrics['node_f1']:.3f} |"
                    f"node class acc : {metrics['class_acc']:.3f}, f1 : {metrics['class_f1']:.3f} ")

            elif is_finetuning :
                val_loss, metrics = evaluate_epoch_finetuning(model, val_loader, loss_fn, device)
                per_channel = " | ".join(f"{key} : {value:.3f}"
                                         for key, value in metrics.items() if key.startswith('ap '))
                line = (f"epoch {epoch} : train loss: {train_loss:.4f} | val loss : {val_loss:.4f} | "
                        f"mAP : {metrics['mAP']:.4f} | {per_channel}")

            else :
                val_loss, metrics = evaluate_epoch(model, val_loader, meta, labels, loss_fn, device)
                line = (f"epoch {epoch} : train loss: {train_loss:.4f} | val loss : {val_loss:.4f}, precision : {metrics['precision']:.4f}, recall : {metrics['recall']:.4f}, f1 : {metrics['f1']:.4f}, f2 : {metrics['f2']:.4f}, rmse : {metrics['rmse']:.4f}, tp : {metrics['tp']}, fp : {metrics['fp']}, fn : {metrics['fn']}")
        
        tqdm.write(line) 
        is_best = logger.log_epoch(epoch, train_loss, val_loss, metrics)
        logger.save_checkpoint(model, epoch, is_best, n_features=n_features, window_size=window_size,
            scaler_mean = meta['scaler'].mean_,
            scaler_scale = meta['scaler'].scale_
            )

    logger.finish()

if __name__ == "__main__":
    main()
