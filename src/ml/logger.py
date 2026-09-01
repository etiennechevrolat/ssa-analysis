"""
Journalisation des entrainements : métriques, checkpoints et W&B
"""

from __future__ import annotations
import torch
import wandb
import numpy as np 
import shutil 

from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf
import json 


class RunLogger:
    def __init__(self, cfg, run_dir):
        self.cfg=cfg
        self.run_dir = Path(run_dir)
        self.chekpoint_dir = self.run_dir / 'checkpoints'
        self.chekpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        ## Le dossier par defaut est horodate a la seconde, donc unique. Mais des qu'on force
        ## hydra.run.dir sur un chemin fixe -- un balayage de folds, typiquement -- relancer
        ## reecrit dans le meme dossier : on archive l'ancien run au lieu de le melanger.
        self._archive_previous_run()

        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_metrics = {}

        self.use_wandb = cfg.wandb.enabled
        if self.use_wandb : 
            wandb.init(
                project = cfg.wandb.project,
                mode = cfg.wandb.mode,
                dir=str(self.run_dir),
                config=OmegaConf.to_container(cfg, resolve=True)
            )

    def _archive_previous_run(self):
        """Deplace les artefacts d'un run precedent au lieu de les melanger aux nouveaux.

        Sans ca, metrics.jsonl -- ouvert en append -- contenait les epochs des deux runs a la
        suite (0..N puis 0..M), ce qui rend les courbes illisibles et l'agregation fausse ;
        best.pt et last.pt, eux, etaient ecrases sans un mot. On archive plutot que d'effacer :
        le run precedent est peut-etre celui qu'on voulait garder.
        """
        anciens = [p for p in (self.metrics_path,
                               self.chekpoint_dir / "last.pt",
                               self.chekpoint_dir / "best.pt") if p.exists()]
        if not anciens:
            return

        date = datetime.fromtimestamp(max(p.stat().st_mtime for p in anciens))
        archive = self.run_dir / f"run_precedent_{date:%Y-%m-%d_%H-%M-%S}"
        ## l'horodatage est celui des fichiers archives, a la seconde : deux archivages
        ## rapproches viseraient le meme dossier et shutil.move ecraserait la premiere archive.
        suffixe = 2
        while archive.exists():
            archive = self.run_dir / f"run_precedent_{date:%Y-%m-%d_%H-%M-%S}_{suffixe}"
            suffixe += 1
        (archive / "checkpoints").mkdir(parents=True)
        for p in anciens:
            shutil.move(str(p), str(archive / p.relative_to(self.run_dir)))

        print(f"[logger] {self.run_dir} contenait deja un run : {len(anciens)} fichier(s) "
              f"deplace(s) dans {archive.name}/. Le nouveau run repart a vide.")

    def watch(self, model):
        if self.use_wandb and self.cfg.wandb.watch:
            wandb.watch(model, log="all", log_freq=100) ## tous les 100 batchs

    @staticmethod
    def _criterion(val_loss, metrics):
        """Quantité à minimiser pour sélectionner le meilleur checkpoint.
        La val loss ne convient pas aux tâches supervisées ici : elle diverge par
        surapprentissage (BCE/CE sur des classes très déséquilibrées) alors que les
        métriques de détection/classification continuent de progresser.
        """
        if "f2" in metrics:  # localizer et finetuning DORIS
            return -metrics["f2"]
        if "node_f1" in metrics:  # classifier
            return -(metrics["node_f1"] + metrics["class_f1"]) / 2
        return val_loss  # pretrain : reconstruction, la val loss est le bon critère

    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Enregistre une epoch, et renvoie True si c'est le meilleur checkpoint."""

        row = {"epoch" :epoch, "train loss" : train_loss, "val loss" : val_loss}
        row.update({f"val {k}" : float(v) for k,v in metrics.items() if np.ndim(v) == 0})
        # un JSON par ligne
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if self.use_wandb:
            wandb.log(row)

        criterion = self._criterion(val_loss, metrics)
        is_best = criterion < self.best_loss
        if is_best :
            self.best_loss, self.best_epoch = criterion, epoch
            ## conservees pour l'apres-boucle : la figure de detection doit etre tracee au
            ## seuil de l'epoch RETENUE, pas a celui de la derniere.
            self.best_metrics = dict(metrics)
        return is_best

    def save_checkpoint(self, model, epoch, is_best = False, **info):
        """
        Ecris last.pt et le copie en best.pt si l'époque est la meilleure
        On sauvegarde le state_dict (dict de tenseurs model.state_dict) avec torch.save(model)
        """
        last = self.chekpoint_dir / "last.pt"

        payload = {
            "epoch" : epoch, 
            "model_state": model.state_dict(),
            "config" : OmegaConf.to_container(self.cfg, resolve=True), 
            **info, 
            }
        if hasattr(model, "encoder_state_dict"):
            payload["encoder_state"] = model.encoder_state_dict()

        torch.save(payload, last)
        if is_best:
            print(f" New best checkpoint saved in {self.chekpoint_dir / "best.pt"}" )
            shutil.copyfile(last, self.chekpoint_dir/ "best.pt")

    def finish(self):
        print(f"[logger] meilleure val loss =  {self.best_loss:.4f} pour l'epoch {self.best_epoch}")
        print(f"[logger] artefacts : {self.run_dir}")
        if self.use_wandb : 
            wandb.finish()

    
        