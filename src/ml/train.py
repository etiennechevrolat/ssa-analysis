### SETUP 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import os 
from pathlib import Path

import hydra 
from omegaconf import DictConfig


from ml.datahandler import load_splid_objects
from ml.dataset import make_loaders
from ml.model import NaiveBaseLine



def find_root(marker: str = "pyproject.toml"):
    p = Path.cwd().resolve()
    for parent in (p, *p.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"{marker} introuvable en remontant depuis {p}")

ROOT = find_root()

DATA_DIR  = os.path.join(ROOT, 'data', 'raw', 'splid_dataset')


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
        x = time_series_batch.to(device)
        y = labels.to(device)

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels.size(0)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
    avg_loss = running_loss / (max(total, 1))
    return avg_loss

@torch.no_grad()
def evaluate_epoch(
    model,
    data_loader,
    loss_fn, 
    device
    ):
    model.eval()
    running_loss = 0.0
    total=0
    for time_series_batch, labels in data_loader:
        x = time_series_batch.to(device)
        y = labels.to(device)

        pred = model(x)

        loss = loss_fn(pred, y)
        running_loss+= float(loss.item()) * labels.size(0)
       

        total += labels.size(0)
    avg_loss = running_loss / (max(total, 1))
    return avg_loss

@hydra.main(version_base=None, config_path="../../configs/ml", config_name="config")
def main(cfg : DictConfig):

    seed = torch.manual_seed(cfg.seed)
    device = cfg.train.device
    data_dir = os.path.join(DATA_DIR, 'training')
    labels_dir = os.path.join(DATA_DIR, 'train_label.csv')

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
        val_split=cfg.data.val_split)

    model = NaiveBaseLine(
        cfg.model.n_features,
        cfg.model.window_size, 
        cfg.model.n_outputs)
    model.to(cfg.train.device)

    params = model.parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.train.lr)

    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(cfg.train.epochs):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va = evaluate_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch} : train {tr:.4f} val {va:.4f}")

if __name__ == "__main__":
    main()
