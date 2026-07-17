### SETUP 
import torch
from torch import nn
from torch.utils.data import DataLoader


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
    correct = 0
    for time_series_batch, labels in data_loader:
        # time_series_batch: (B,F,W), labels (B,2)
        x = time_series_batch.to(device)
        y = labels.to(device)
        print(labels)
        optimizer.zero_grad()

        pred = model(x)

        correct += int((pred == labels).sum().item())
        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels.size(0)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct/total
    return avg_loss, accuracy


@torch.no_grad() ## décorateur, applique torch.no_grad(evaluate_epoch(...)) et coupe le suivi des gradients.
def evaluate_epoch(
    model,
    data_loader,
    loss_fn, 
    device
    ):
    model.eval() ## différents de torch.no_grad, change le comportement de certaines couches : Dropout, BatchNorm.
    running_loss = 0.0
    total=0
    correct = 0 
    for time_series_batch, labels in data_loader:
        x = time_series_batch.to(device)
        y = labels.to(device)

        pred = model(x)
        correct += int((pred == labels).sum().item())
        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels.size(0)
    

        total += labels.size(0)
    avg_loss = running_loss / total
    accuracy = correct/total
    return avg_loss, accuracy

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
    n_features = len(meta["features_cols"])
    window_size  = cfg.history + cfg.future +1 

    model = build_model(cfg.model, n_features, window_size)
    model.to(device)

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr = cfg.train.lr)

    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.train.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch} : train loss/acc : {train_loss:.4f} | {train_acc} val loss/acc : {val_loss:.4f}  | {val_acc}")

if __name__ == "__main__":
    main()
