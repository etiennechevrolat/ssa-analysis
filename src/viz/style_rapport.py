"""Conventions de tracé des figures de rapport.

On reprend l'idiome deja en place dans le depot (src/viz/time_series.py et
maneuver_detection/discrete_kalman_filter.py) plutot que d'en inventer un autre :

  - serie de parametre orbital : vert, lw 1.2
  - manoeuvres : traits verticaux rouges pointilles, lw 0.7, alpha 0.5
  - statistique / NIS : tab:blue, lw 0.8
  - seuil : gris, tirets, lw 0.8
  - grille : tirets, lw 0.3, alpha 0.5 ; ticks a 7 pt, labels d'axe a 9 pt
  - panneaux larges (12 de large pour 4 de haut), pas de marge en x

`_axis_label` et `LABELS` viennent directement de time_series pour que les
intitules d'axes soient les memes d'une figure a l'autre.
"""

import matplotlib as mpl
import matplotlib.ticker as mticker

from viz.time_series import LABELS, _axis_label  # noqa: F401

SERIE = "green"
MANOEUVRE = "red"
STAT = "tab:blue"
SEUIL = "grey"
MODELE = "black"


def applique():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "legend.framealpha": 0.7,
    })


def grille(ax):
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)
    ax.tick_params(labelsize=7)
    ## ScalarFormatter seulement sur un axe lineaire : sur une echelle log il
    ## etiquette 0,01 en "0.0" et rend l'axe illisible.
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.margins(x=0)


def manoeuvres(ax, dates, etiquette="Manoeuvres"):
    """Verite terrain : traits verticaux pointilles."""
    for j, m in enumerate(dates):
        ax.axvline(m, color="black", lw=0.7, ls=":", alpha=0.8,
                   label=etiquette if j == 0 else None)


def detections(ax, dates, etiquette="Détections"):
    """Sorties du detecteur : traits verticaux rouges pleins."""
    for j, m in enumerate(dates):
        ax.axvline(m, color=MANOEUVRE, lw=0.8, alpha=0.8,
                   label=etiquette if j == 0 else None)
