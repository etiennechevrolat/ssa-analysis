import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from ml.inference import load_checkpoint, load_spacetrack_features, reconstruct_window

def load_mae_and_data(ckpt_id, base_path=None):
    if base_path is None:
        base_path = Path.cwd()

    ckpt_path = base_path / "outputs" / "ml" / "pretrain" / ckpt_id / "checkpoints" / "best.pt"
    data_dir = base_path / "data" / "raw" / "spacetrack"

    mae, cfg, mean, scale = load_checkpoint(ckpt_path, device="cpu")
    
    per_obj, _, feature_cols = load_spacetrack_features(
        data_dir, cfg.data.dataset, mean, scale, return_stats=True
    )
    per_obj = {norad: X for norad, X in per_obj.items() if len(X) >= cfg.data.window_size}
    
    return mae, cfg, mean, scale, per_obj, feature_cols

# Chargement du modèle et des données une bonne fois pour toutes
mae, cfg, mean, scale, per_obj, feature_cols = load_mae_and_data(ckpt_id="2026-08-27_11-42-43")
window_size = cfg.data.window_size
patch_size = cfg.model.patch_size
stats = {norad: (mean, scale) for norad in per_obj}

def plot_object_windows(norad, n_windows=3, to_plot=["sma_local"]):
    X = per_obj[norad]
    starts = np.linspace(0, len(X) - window_size, n_windows).astype(int)
    
    for window_start in starts:
        x = X[window_start : window_start + window_size].T
        recon, masked = reconstruct_window(mae, x, device="cpu", seed=0)

        mu, sigma = stats[norad]
        x_phys = x.T * sigma + mu
        recon_phys = recon.T * sigma + mu
        t = np.arange(window_start, window_start + window_size)

        fig, axes = plt.subplots(len(to_plot), 1, figsize=(14, 2.6 * len(to_plot)), sharex=True)
        if len(to_plot) == 1:
            axes = [axes]

        for ax, name in zip(axes, to_plot):
            col = feature_cols.index(name)
            for p in masked:
                ax.axvspan(t[p * patch_size], t[min((p + 1) * patch_size, window_size - 1)], color="grey", alpha=0.18, lw=0)
            
            pred = np.full(window_size, np.nan)
            for p in masked:
                pred[p * patch_size : (p + 1) * patch_size] = recon_phys[p * patch_size : (p + 1) * patch_size, col]

            ax.plot(t, x_phys[:, col], color="black", lw=1.4, label="original")
            ax.plot(t, pred, color="tab:red", lw=1.6, label="reconstruit")
            ax.set_ylabel(name)

        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"norad {norad} — patchs masqués grisés")
        axes[-1].set_xlabel("TimeIndex (TLE)")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_object_windows(norad=49217, n_windows=4, to_plot=["sma_local"])

