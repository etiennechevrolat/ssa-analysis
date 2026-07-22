### SETUP 
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import hydra 
from omegaconf import DictConfig


from ml.datahandler import load_splid_objects
from ml.dataset import make_loaders
from ml.model import build_model


def train_one_epoch(
        model : nn.Module,
        data_loader : DataLoader,
        loss_fn : nn.Module, 
        optimizer : torch.optim.Optimizer,
        device : torch.device
    ):
    model.train()
    running_loss = 0.0
    total = 0

    for time_series_batch, labels in data_loader:
        # time_series_batch: (B,F,W), labels (B,2)
        x = time_series_batch.to(device)
        y = labels.to(device)

        optimizer.zero_grad()

        pred = model(x)


        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels.size(0)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
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
    for time_series_batch, labels_batch in data_loader:
        x = time_series_batch.to(device)
        y = labels_batch.to(device)

        pred = model(x) ## format (Batch_size, 2)

        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels_batch.size(0)
        total += labels_batch.size(0)
        
        all_probs.append(torch.sigmoid(pred).cpu().numpy())

    ## On met les probas au format attendu pour extract_events (L,2) avec L longueur d'UN objet : découpage de la data par objet
    all_probs = np.concatenate(all_probs, axis=0) ## (N,2)
    val_ids = meta['val_ids']
    objects_lengths = [len(meta['per_obj'][oid][0]) for oid in val_ids]

    assert(np.sum(objects_lengths) == len(all_probs)) ## on s'assure de la correspondance des tailles.

    seq_per_obj = np.split(all_probs, np.cumsum(objects_lengths)[-1], axis=0) ## listes (L1,2) (L2, 2) ... (Li, 2) de probas par objet

    

    pred_events = {oid : extract_events(seq) for oid, seq in zip(val_ids, seq_per_obj)}
    gt_events = gt_events_from_labels(labels, val_ids) ## labels est ici le dataframe brut rendu par load_splid_objects

    metrics = evaluate_predictions(gt_events, pred_events) ## renvoie dictionnaire des métriques de performance
    
    avg_loss = running_loss / total
    return avg_loss, metrics

@hydra.main(version_base=None, config_path="../../configs/ml", config_name="config") 
## hydra prend main en point d'entrée, et main() est transformé en wrapper qui récupère et construit l'objet cfg, et enfin applique main(cfg).
def main(cfg : DictConfig):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Recup des données et creation des dataloaders
    objects, labels = load_splid_objects(
        cfg.data.data_dir, 
        cfg.data.labels_dir,
        )
    
    train_loader, val_loader, meta = make_loaders(
        objects, 
        labels,
        batch_size=cfg.data.batch_size,
        history=cfg.data.history, 
        future=cfg.data.future,
        val_split=cfg.data.val_split,
        seed= cfg.seed)
    ## On calcule le nombre de features considérées et la taille de la fenêtre temporelle
    n_features = len(meta["feature_cols"])
    window_size  = cfg.data.history + cfg.data.future +1 

    model = build_model(cfg.model, n_features=n_features, window_size=window_size)
    model.to(device)

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr = cfg.train.lr)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.train.epochs):
        train_loss= train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, metrics = evaluate_epoch(model, val_loader, meta, labels, loss_fn, device)
        print(f"epoch {epoch} : train loss: {train_loss:.4f} | val loss : {val_loss:.4f}, precision : {metrics['precision']:.4f}, recall : {metrics['recall']:.4f}, f1 : {metrics['f1']:.4f}, f2 : {metrics['f2']:.4f}, rmse : {metrics['rmse']:.4f}, tp : {metrics['tp']}, fp : {metrics['fp']}, fn : {metrics['fn']}")

if __name__ == "__main__":
    main()
