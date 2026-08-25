"""
Script pour visualiser la reconstruction d'un MAE (Masked Autoencoder)
sur des données SpaceTrack (TLE).
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional

def load_mae_and_data(
    ckpt_id: str,
    base_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
) :
    if base_path is None:
        base_path = Path.cwd()

    ckpt_path = Path(
        os.path.join(
            base_path, "outputs", "ml", "pretrain", ckpt_id, "checkpoints", "best.pt"
        )
    )
    data_dir = Path(os.path.join(base_path, "data", "raw", "spacetrack"))

    from ml.inference import load_checkpoint, load_spacetrack_features

    mae, cfg, mean, scale = load_checkpoint(ckpt_path, device)
    return mae, cfg, mean, scale

# --- 2. Préparation des données par objet ---
def load_object_data(
    data_dir: Path,
    cfg: object,
    mean: np.ndarray,
    scale: np.ndarray,
    min_window_size: int = 256,
) :
    """
    Charge et filtre les données SpaceTrack par objet, en ne gardant que ceux
    avec assez de TLE pour former une fenêtre de taille `min_window_size`.
    """
    from ml.inference import load_spacetrack_features

    per_obj, _, feature_cols = load_spacetrack_features(
        data_dir, cfg.data.dataset, mean, scale, return_stats=True
    )
    # Filtrer les objets avec assez de TLE
    per_obj = {norad: X for norad, X in per_obj.items() if len(X) >= min_window_size}
    return per_obj, feature_cols

def plot_reconstruction(
    mae: torch.nn.Module,
    per_obj: Dict[int, np.ndarray],
    stats: Dict[int, Tuple[np.ndarray, np.ndarray]],
    feature_cols: List[str],
    norad: int,
    window_start: int,
    to_plot: List[str] = ["sma_local"],
    window_size: int = 256,
    patch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
):
    """
    Trace la reconstruction du MAE pour un objet donné, en comparant
    les séries originales et reconstruites. Les patchs masqués sont grisés.
    """
    from ml.inference import reconstruct_window

    X = per_obj[norad]
    x = X[window_start : window_start + window_size].T  # (F, W) normalisé
    recon, masked = reconstruct_window(mae, x, device=device, seed=seed)

    mu, sigma = stats[norad]  # Stats de cet objet
    x_phys, recon_phys = x.T * sigma + mu, recon.T * sigma + mu  # Unités physiques
    t = np.arange(window_start, window_start + window_size)

    fig, axes = plt.subplots(len(to_plot), 1, figsize=(14, 2.6 * len(to_plot)), sharex=True)
    if len(to_plot) == 1:
        axes = [axes]  # Pour itérer même avec un seul subplot

    for ax, name in zip(axes, to_plot):
        col = feature_cols.index(name)
        # Griser les patchs masqués
        for p in masked:
            ax.axvspan(
                t[p * patch_size],
                t[min((p + 1) * patch_size, window_size - 1)],
                color="grey",
                alpha=0.18,
                lw=0,
            )
        # Tracer la reconstruction (uniquement sur les patchs masqués)
        pred = np.full(window_size, np.nan)
        for p in masked:
            sl = slice(p * patch_size, (p + 1) * patch_size)
            pred[sl] = recon_phys[sl, col]

        ax.plot(t, x_phys[:, col], color="black", lw=1.4, label="original")
        ax.plot(t, pred, color="tab:red", lw=1.6, label="reconstruit (patchs masqués)")
        ax.set_ylabel(name)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"norad {norad} — patchs masqués grisés")
    axes[-1].set_xlabel("TimeIndex (TLE)")
    fig.tight_layout()
    plt.show()

def plot_object_windows(
    mae: torch.nn.Module,
    per_obj: Dict[int, np.ndarray],
    stats: Dict[int, Tuple[np.ndarray, np.ndarray]],
    feature_cols: List[str],
    norad: int,
    n_windows: int = 3,
    to_plot: List[str] = ["sma_local"],
    window_size: int = 256,
    patch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    seed: int = 0):
    """
    Trace `n_windows` fenêtres réparties sur toute la série d'un objet.
    """
    starts = np.linspace(0, len(per_obj[norad]) - window_size, n_windows).astype(int)
    for start in starts:
        plot_reconstruction(
            mae,
            per_obj,
            stats,
            feature_cols,
            norad,
            window_start=int(start),
            to_plot=to_plot,
            window_size=window_size,
            patch_size=patch_size,
            device=device,
            seed=seed,
        )


if __name__ == "__main__":
    #Charger le modèle et les données
    mae, cfg, mean, scale = load_mae_and_data(ckpt_id="2026-08-25_15-09-46")
    window_size = cfg.data.window_size
    patch_size = cfg.model.patch_size

    # Charger les données par objet
    per_obj, feature_cols = load_object_data(
        data_dir=Path(os.path.join(Path.cwd(), "data", "raw", "spacetrack")),
        cfg=cfg,
        mean=mean,
        scale=scale,
        min_window_size=window_size,
    )
    stats = {norad: (mean, scale) for norad in per_obj}  # À adapter !

    # Tracer pour 10 objets aléatoires
    import random
    for norad in random.sample(list(per_obj), 10):
        plot_object_windows(
            mae,
            per_obj,
            stats,
            feature_cols,
            norad,
            n_windows=3,
            to_plot=["sma_local", 'k', 'h', 'p', 'q'],
            window_size=window_size,
            patch_size=patch_size,
        )


