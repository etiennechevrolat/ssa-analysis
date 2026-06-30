"""Tracés de séries temporelles des paramètres orbitaux.

Les données sont lues depuis les fichiers .parquet de data/raw via
`storage.parquet.load_history`, qui renvoie un DataFrame polars trié par epoch
et dédupliqué (une ligne par epoch, dernière creation_date conservée).

Schéma des .parquet : norad, object_name, epoch, creation_date, rev_at_epoch,
inclination, raan, arg_perigee, mean_anomaly, mean_motion, eccentricity, sma,
apogee, perigee, period, velocity.
"""

import polars as pl
import matplotlib.pyplot as plt

# Libellé lisible + unité pour chaque paramètre, pour des axes propres.
LABELS = {
    "sma":          ("Demi-grand axe", "km"),
    "eccentricity": ("Excentricité", "-"),
    "inclination":  ("Inclination", "°"),
    "raan":         ("RAAN", "°"),
    "arg_perigee":  ("Argument of Perigee", "°"),
    "mean_anomaly": ("Mean Anomaly", "°"),
    "mean_motion":  ("Mean Motion", "rev./day"),
    "period":       ("Période", "s"),
    "velocity":     ("Vitesse", "m/s"),
    "rev_at_epoch": ("Révolution à l'epoch", ""),
    "bstar":        ("B*", "1/m"),
}

# Paramètres pour lesquels on applique un ylim par percentile [1,99]
# afin d'éviter que des outliers TLE écrasent l'échelle.
PERCENTILE_CLIP = {"eccentricity", "inclination", "bstar"}


# Utilisation :
# tout l'historique de l'objet :
#load_history(path, 39634, params)
# seulement mars–avril 2024 :
# load_history(path, 39634, params, start=datetime(2024,3,1), end=datetime(2024,5,1))

def load_history(path, norad, params, start= None, end= None, time_col='epoch'):
    # Charge une ligne pour un id / epoch donnée

    available = pl.scan_parquet(path).collect_schema().names()
    
    q = pl.scan_parquet(path).filter(pl.col("norad") == norad)

    if "creation_date" in available:
        q = q.sort("creation_date").unique(subset=[time_col], keep="last" )

    q = q.select([time_col, *params ]).sort(time_col)
    if start is not None : 
        q= q.filter(pl.col(time_col)>= start)
    if end is not None: 
        q= q.filter(pl.col(time_col)<= end)
    return q.collect()


def _axis_label(param):
    name, unit = LABELS.get(param, (param, ""))
    return f"{name} [{unit}]" if unit else name



def extract_series(history, params, time_col="epoch"):
    """Sépare un historique tidy en séries temporelles {param: (t, y)},
    `history` est le DataFrame polars renvoyé par `load_history`
    (colonnes : time_col + params, trié par time_col).
    """
    if isinstance(params, str):
        params = [params]
    t = history[time_col].to_list()
    return {p: (t, history[p].to_list()) for p in params}


def plot_time_series(path, norad, params, start=None, end=None, time_col="epoch", ncols=3, ylim=None):
    """Trace l'évolution de `params` pour un satellite — grille ncols colonnes.

    Args:
        ylim: dict optionnel {param: (lo, hi)} pour forcer l'échelle d'un paramètre.
              Ex : ylim={"eccentricity": (0, 0.002), "inclination": (53, 54)}

    Exemple :
        fig = plot_time_series(path, 45185, ["eccentricity", "inclination", "raan",
                               "mean_anomaly", "mean_motion", "arg_perigee"],
                               ylim={"eccentricity": (0, 0.001)})
    """
    import math
    import numpy as np
    import matplotlib.ticker as mticker

    if isinstance(params, str):
        params = [params]
    ylim = ylim or {}

    history = load_history(path, norad, params, start, end, time_col)
    if history.is_empty():
        raise ValueError(f"Aucune donnée pour norad={norad} dans {path}")

    series = extract_series(history, params, time_col)

    ncols = min(ncols, len(params))
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        t, y = series[p]
        vals = np.array([v if v is not None else np.nan for v in y], dtype=float)

        if p in ylim:
            lo, hi = ylim[p]
            vals = np.clip(vals, lo, hi)
        elif p in PERCENTILE_CLIP:
            finite = vals[np.isfinite(vals)]
            if len(finite):
                lo, hi = np.percentile(finite, [1, 99])
                vals = np.clip(vals, lo, hi)

        ax.plot(t, vals, color="green", lw=1.2)
        ax.set_ylabel(_axis_label(p), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))

    for i in range(len(params), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(f"NORAD {norad} — {len(history)} points", fontsize=11)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_compare(path, norads, params, start=None, end=None, time_col="epoch", ncols=3, ylim=None):
    """Superpose plusieurs satellites sur une grille de sous-graphes (un par paramètre).

    Args:
        ylim: dict optionnel {param: (lo, hi)} pour forcer l'échelle d'un paramètre.
    """
    import math
    import numpy as np
    import matplotlib.ticker as mticker

    if isinstance(params, str):
        params = [params]
    ylim = ylim or {}
    norads = list(dict.fromkeys(norads))  # déduplique en préservant l'ordre

    ncols = min(ncols, len(params))
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    
    # Chargement unique de chaque satellite (évite de relire les parquet 2×)
    histories = []  # liste de (norad, t, {param: vals_array})
    all_raw: dict[str, list] = {p: [] for p in params}
    for norad in norads:
        try:
            history = load_history(path, norad, params, start, end, time_col)
        except Exception:
            continue
        if history.is_empty():
            continue
        t = history[time_col].to_list()
        cols = {}
        for p in params:
            vals = np.array([v if v is not None else np.nan for v in history[p].to_list()], dtype=float)
            cols[p] = vals
            all_raw[p].extend(vals[np.isfinite(vals)].tolist())
        histories.append((norad, t, cols))

    # Bornes de clip à partir des valeurs déjà chargées
    clip_bounds: dict[str, tuple] = {}
    for p in params:
        if p in ylim:
            clip_bounds[p] = ylim[p]
        elif p in PERCENTILE_CLIP and all_raw[p]:
            lo, hi = np.percentile(np.array(all_raw[p]), [1, 99])
            clip_bounds[p] = (lo, hi)

    for norad, t, cols in histories:
        for i, p in enumerate(params):
            ax = axes[i // ncols][i % ncols]
            vals = cols[p]
            if p in clip_bounds:
                vals = np.clip(vals, *clip_bounds[p])
            ax.plot(t, vals, lw=0.7, alpha=0.8, label=f"NORAD {norad}")

    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        ax.set_ylabel(_axis_label(p), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))

    for i in range(len(params), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()

    # Légende unique sous la figure (évite la répétition par sous-graphe)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(handles) <= 20:
        ncols_leg = min(len(handles), 6)
        fig.legend(handles, labels, loc="lower center", ncol=ncols_leg,
                   fontsize=7, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.02 - 0.03 * math.ceil(len(handles) / ncols_leg)))
    
    return fig


def plot_cluster(
    path,
    norads,
    params,
    cluster_label=None,
    start=None,
    end=None,
    time_col="epoch",
    ncols=3,
):
    """Superpose tous les satellites de `norads` en gris, avec la moyenne du
    cluster en vert — grille ncols colonnes, style Guimaraes et al.

    Exemple :
        fig = plot_cluster(path, norads_list, ["eccentricity", "inclination",
                           "raan", "mean_anomaly", "mean_motion", "arg_perigee"],
                           cluster_label="Cluster 62")
        fig.savefig("cluster62.png", dpi=150)
    """
    import math
    import numpy as np
    import matplotlib.ticker as mticker

    if isinstance(params, str):
        params = [params]
    if cluster_label is None:
        cluster_label = f"Cluster ({len(norads)} sat.)"

    ncols = min(ncols, len(params))
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )
    # axes plats dans l'ordre des params
    flat_axes = [axes[i // ncols][i % ncols] for i in range(len(params))]
    # masquer les axes excédentaires
    for i in range(len(params), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    # Accumule les séries de tous les satellites pour calculer la moyenne
    # Structure : {param: list of (t_array, y_array)}
    all_series: dict[str, list] = {p: [] for p in params}

    for norad in norads:
        try:
            hist = load_history(path, norad, params, start, end, time_col)
        except Exception:
            continue
        if hist.is_empty():
            continue
        t = hist[time_col].to_list()
        for p in params:
            all_series[p].append((t, hist[p].to_list()))

    # Trace individuel gris + calcul de la moyenne sur grille temporelle commune
    for ax, p in zip(flat_axes, params):
        ax.set_ylabel(_axis_label(p), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

        if not all_series[p]:
            continue

        # Traces individuelles
        for t, y in all_series[p]:
            ax.plot(t, y, color="lightgray", lw=0.5, alpha=0.7, zorder=1)

        # Moyenne : interpolation sur une grille commune
        # On convertit les epochs en timestamps numériques pour np.interp
        all_t_num = sorted({
            epoch.timestamp() if hasattr(epoch, "timestamp") else float(epoch)
            for t, _ in all_series[p] for epoch in t
        })
        if len(all_t_num) < 2:
            continue

        grid = np.array(all_t_num)
        interp_matrix = []
        for t, y in all_series[p]:
            t_num = np.array([
                e.timestamp() if hasattr(e, "timestamp") else float(e) for e in t
            ])
            y_arr = np.array(y, dtype=float)
            # Interpoler uniquement dans la plage couverte par ce satellite
            mask = (grid >= t_num[0]) & (grid <= t_num[-1])
            if mask.sum() < 2:
                continue
            yi = np.interp(grid[mask], t_num, y_arr)
            row = np.full(len(grid), np.nan)
            row[mask] = yi
            interp_matrix.append(row)

        if interp_matrix:
            mat = np.array(interp_matrix)
            mean_y = np.nanmean(mat, axis=0)
            # Reconstruire les datetimes pour l'axe x
            import datetime
            mean_t = [datetime.datetime.fromtimestamp(ts) for ts in grid]
            valid = ~np.isnan(mean_y)
            ax.plot(
                np.array(mean_t)[valid], mean_y[valid],
                color="green", lw=1.5, zorder=3,
                label=f"{cluster_label} Mean",
            )

        # Autoscale serré sur les données réelles
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="y")
        ax.autoscale(axis="y", tight=False)
        ax.margins(y=0.05)

        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    return fig




from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "graphs"


def save_fig(fig, name, subdir=None, dpi=150):
    """Enregistre une figure dans OUTPUT (data/graphs) et la ferme.
    Args:
        fig:     Figure matplotlib à sauvegarder.
        name:    Nom du fichier sans extension (ex. "starlink_45185_sma").
        subdir:  Sous-dossier optionnel dans OUTPUT (ex. "time_series").
        dpi:     Résolution en points par pouce.

    Exemple :
        save_fig(fig, f"norad_{norad}_orbital")
        save_fig(fig, "cluster62", subdir="clusters")
    """
    dest = OUTPUT / subdir if subdir else OUTPUT
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure écrite dans {path}")

