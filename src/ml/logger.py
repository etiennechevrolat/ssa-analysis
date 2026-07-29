"""
Journalisation des entrainements : métriques, checkpoints et W&B
"""

from __future__ import annotations
import torch
import wandb
import numpy as np 
import shutil 

from pathlib import Path

from omegaconf import OmegaConf
import json 


class RunLogger:
    def __init__(self, cfg, run_dir):
        self.cfg=cfg
        self.run_dir = Path(run_dir)
        self.chekpoint_dir = self.run_dir / 'checkpoints'
        self.chekpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")
        self.best_epoch = None

        self.use_wandb = cfg.wandb.enabled
        if self.use_wandb : 
            wandb.init(
                project = cfg.wandb.project,
                mode = cfg.wandb.mode,
                dir=str(self.run_dir),
                config=OmegaConf.to_container(cfg, resolve=True)
            )

    def watch(self, model):
        if self.use_wandb and self.cfg.wandb.watch:
            wandb.watch(model, log="all", log_freq=100) ## tous les 100 batchs

    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Enregistre une epoch, et renvoie True si c'est la meilleure val_loss"""

        row = {"epoch" :epoch, "train loss" : train_loss, "val loss" : val_loss}
        row.update({f"val {k}" : float(v) for k,v in metrics.items() if np.ndim(v) == 0})
        # un JSON par ligne
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

        if self.use_wandb:
            wandb.log(row)

        is_best = val_loss < self.best_loss 
        if is_best : 
            self.best_loss, self.best_epoch = val_loss, epoch
        return is_best

    def save_checkpoint(self, model, epoch, is_best = False, **info):

        """
        Ecris last.pt et le copie en best.pt si l'époque est la meilleure
        On sauvegarde le state_dict (dict de tenseurs model.state_dict) avec torch.save(model)
        """
        last = self.chekpoint_dir / "last.pt"

        torch.save({
            "epoch" : epoch, 
            "model_state": model.state_dict(),
            "config" : OmegaConf.to_container(self.cfg, resolve=True), 
            **info, 
            }, last)

        if is_best:
            shutil.copyfile(last, self.chekpoint_dir/ "best.pt")

    def finish(self):
        print(f"[logger] meilleure val loss =  {self.best_loss:.4f} epoch {self.best_epoch})")
        print(f"[logger] artefacts : {self.run_dir}")
        if self.use_wandb : 
            wandb.finish()

    
        