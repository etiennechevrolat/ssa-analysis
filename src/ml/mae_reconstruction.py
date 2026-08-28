from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ml.inference import load_checkpoint, load_spacetrack_features, reconstruct_window


def load_mae_and_data(ckpt_id, base_path=None):
    """Charge le modele et les features normalisees 'dataset' (avant RevIN par fenetre).

    Deux normalisations distinctes et independantes entrent en jeu :
      - normalisation dataset (globale ou par objet), appliquee ICI sur per_obj. Son inverse
        est stats[norad] = (mu, sigma) : en mode global tous les objets partagent le meme
        (mu, sigma) ; en mode par objet (scaler_kind='per_obj') chacun a le sien. C'est
        load_spacetrack_features qui decide, via mean/scale issus du checkpoint : c'est
        la SEULE source de verite pour repasser en unites physiques, quel que soit le mode.
      - RevIN par fenetre, interne au modele (TimeSeriesMAE._instance_stats), applique ET
        annule entierement a l'interieur de reconstruct_window. L'appelant n'a jamais besoin
        d'y toucher : reconstruct_window renvoie deja la reconstruction dans l'espace de la
        normalisation dataset ci-dessus, prete a etre multipliee par stats[norad].
    """
    ## racine du repo deduite de l'emplacement de ce module (src/ml/ -> ../..), et non du
    ## cwd : un notebook demarre son kernel dans son propre dossier, un script lance depuis
    ## la racine non, et un Path.cwd() relatif casse dans l'un des deux cas.
    if base_path is None:
        base_path = Path(__file__).resolve().parents[2]
    base_path = Path(base_path).resolve()

    ckpt_path = base_path / "outputs" / "ml" / "pretrain" / ckpt_id / "checkpoints" / "best.pt"
    data_dir = base_path / "data" / "raw" / "spacetrack"

    mae, cfg, mean, scale = load_checkpoint(ckpt_path, device="cpu")

    per_obj, stats, feature_cols = load_spacetrack_features(
        data_dir, cfg.data.dataset, mean, scale, return_stats=True
    )
    per_obj = {norad: X for norad, X in per_obj.items() if len(X) >= cfg.data.window_size}
    stats = {norad: mu_sigma for norad, mu_sigma in stats.items() if norad in per_obj}

    return mae, cfg, stats, per_obj, feature_cols


def plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad, n_windows=3, to_plot=("sma_local",)):

    window_size = cfg.data.window_size
    patch_size = cfg.model.patch_size

    X = per_obj[norad]
    starts = np.linspace(0, len(X) - window_size, n_windows).astype(int)

    for window_start in starts:
        x = X[window_start : window_start + window_size].T
        recon, masked = reconstruct_window(mae, x, device="cpu", seed=0)

        ## mu, sigma : normalisation dataset de cet objet (globale ou par objet selon le
        ## checkpoint, cf. load_mae_and_data). recon est deja debarrasse du RevIN par fenetre.
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
    mae, cfg, stats, per_obj, feature_cols = load_mae_and_data(ckpt_id="2026-08-28_21-27-33")
    plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad=49217, n_windows=1, to_plot=["sma_local"])
