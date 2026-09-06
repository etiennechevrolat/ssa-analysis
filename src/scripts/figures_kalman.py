"""Figures du rapport sur la detection par filtrage de Kalman.

Ecrites dans rapport/kalman/figures/. Le tableau de metriques est genere en fragment
LaTeX que le .tex appelle par \\input : aucun chiffre recopie a la main.

Le trace suit l'idiome deja en place dans le depot (cf. viz/style_rapport).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maneuver_detection.discrete_kalman_filter import kalman_filter_ordre_1  # noqa: E402
from maneuver_detection.eval_common import agrege, to_days  # noqa: E402
from viz.style_rapport import (  # noqa: E402
    MANOEUVRE, MODELE, SERIE, SEUIL, STAT,
    applique, detections, grille, manoeuvres,
)

FIG = ROOT / "rapport" / "kalman" / "figures"
OUT = ROOT / "outputs" / "eval_doris200"
TLE = OUT / "tle_doris200.parquet"
LABELS = ROOT / "data/parsed/labelled_leo_DORIS/leo_maneuvers_label_augmented_sso_200.csv"
ALPHA = 0.997
SEUIL_CHI2 = chi2.ppf(ALPHA, df=1)

## Les memes objets que la galerie du rapport sur le test statistique, pour que les
## deux methodes se comparent a vue. Canal sma seul : l'optimisation par objet n'a
## ete menee que sur ce canal.
SELECTION = [43609, 41727, 40336, 38257, 39150, 43260, 43619, 47932]

## De la borne haute (une optimisation par objet) a la borne basse (un jeu unique).
NOMS = {"individuel": "params\nindividuels (200)",
        "cluster_brutes": "cluster\nfeatures brutes (8)",
        "cluster_embeddings": "cluster\nembeddings (5)",
        "moyen": "params\nmoyens (1)"}


def serie_nis(norad, var_Q, r, p0, canal="sma"):
    df = (pl.scan_parquet(TLE).filter(pl.col("norad") == norad)
          .select("epoch", canal).sort("epoch").collect())
    _, nis, _ = kalman_filter_ordre_1(df, var_Q=var_Q, r=r, p0=p0, metrique=canal)
    e = df["epoch"].to_numpy()
    return e[1:], df[canal].to_numpy()[1:], np.asarray(nis, float)


def labels():
    lab = pd.read_csv(LABELS)
    lab["epoch"] = pd.to_datetime(lab.epoch, format="mixed", utc=True).dt.tz_localize(None)
    return lab[lab.maneuver_type.isin(("in-track", "radial"))]


def _detendu(t, y):
    return y - pd.Series(y, index=pd.to_timedelta(t, "D")).rolling(
        "30D", center=True, min_periods=12).median().to_numpy()


def _fenetre_lisible(t, vd, largeur, n_min=3, n_max=6):
    interieur = (t > t[0] + 30) & (t < t[-1] - 30)
    ti = t[interieur]
    debuts = np.arange(ti[0], max(ti[-1] - largeur, ti[0] + 1), largeur / 3)
    n_man = np.array([((vd >= d) & (vd < d + largeur)).sum() for d in debuts])
    ok = np.flatnonzero((n_man >= n_min) & (n_man <= n_max))
    d0 = debuts[ok[len(ok) // 2]] if len(ok) else debuts[int(np.argmax(n_man))]
    return d0, interieur


# --------------------------------------------------- 1. le filtre en action

def fig_nis(norad=43437, largeur=360):
    """Residu du demi-grand axe et residu normalise, sur toute la largeur.

    On saute le premier cinquieme de la serie : c'est la mise a poste, dont les
    manoeuvres font plusieurs kilometres alors que le maintien a poste se joue au
    metre.
    """
    opt = pd.read_csv(OUT / "kalman_opt_sma.csv")
    p = opt[(opt.norad == norad) & (opt.ordre == 1)].iloc[0]
    ep, a, nis = serie_nis(norad, p.var_Q, p.r, p.p0)
    t = to_days(ep, ep[0])
    lab = labels()
    v = lab[lab.norad_id == norad].epoch.to_numpy().astype("datetime64[ns]")
    vd = (v - ep[0]).astype(float) / 86400e9

    d0 = t[0] + 0.2 * (t[-1] - t[0])
    m = (t >= d0) & (t <= d0 + largeur)
    det = _detendu(t, a) * 1000.0
    vd = vd[(vd >= d0) & (vd <= d0 + largeur)] - d0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True)
    ax1.plot(t[m] - d0, det[m], lw=1.0, color=SERIE)
    manoeuvres(ax1, vd)
    ax1.set_ylabel("Écart au demi-grand axe [m]", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    grille(ax1)

    ax2.plot(t[m] - d0, nis[m], lw=0.8, color=MODELE, label="NIS $= y^2/S$")
    ax2.axhline(SEUIL_CHI2, color=SEUIL, ls="--", lw=0.8,
                label=rf"seuil $\chi^2_1(\alpha={ALPHA})$ = {SEUIL_CHI2:.2f}")
    manoeuvres(ax2, vd)
    ax2.set_ylabel("NIS", fontsize=9)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("Jours", fontsize=9)
    ax2.set_xlim(0, largeur)
    ax2.legend(loc="upper right", fontsize=8)
    grille(ax2)

    fig.suptitle(f"NORAD {norad} — filtre d'ordre 1, paramètres optimisés", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "kalman_nis_exemple.png")
    plt.close(fig)
    print(f"kalman_nis_exemple.png   (norad {norad}, var_Q={p.var_Q:.3g}, r={p.r:.3g})")


# ------------------------------------------------------- 2. les detections

def fig_detections(largeur=130):
    """Huit objets in-track, aux parametres optimises objet par objet."""
    opt = pd.read_csv(OUT / "kalman_opt_sma.csv")
    opt = opt[opt.ordre == 1].set_index("norad")
    lab = labels()

    n = len(SELECTION)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 1.9 * ((n + 1) // 2)),
                             squeeze=False)
    for ax, norad in zip(axes.ravel(), SELECTION):
        p = opt.loc[norad]
        ep, a, nis = serie_nis(norad, p.var_Q, p.r, p.p0)
        t = to_days(ep, ep[0])
        det = _detendu(t, a) * 1000.0
        v = lab[lab.norad_id == norad].epoch.to_numpy().astype("datetime64[ns]")
        vd = (v - ep[0]).astype(float) / 86400e9

        d0, interieur = _fenetre_lisible(t, vd, largeur)
        m = interieur & (t >= d0) & (t <= d0 + largeur)

        ax.plot(t[m] - d0, det[m], color=SERIE, lw=1.0)
        manoeuvres(ax, vd[(vd >= d0) & (vd <= d0 + largeur)] - d0)
        detections(ax, t[m & (nis > SEUIL_CHI2)] - d0)
        ax.set_title(f"NORAD {norad} — in-track", fontsize=9)
        ax.set_ylabel("Écart au demi-grand axe [m]", fontsize=7.5)
        ax.set_xlim(0, largeur)
        grille(ax)
    for ax in axes.ravel()[-2:]:
        ax.set_xlabel("Jours", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, framealpha=0.7,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout()
    fig.savefig(FIG / "kalman_detections.png")
    plt.close(fig)
    print(f"kalman_detections.png   ({n} objets)")


# ---------------------------------------------------- 3. les 4 configurations

def fig_configs():
    """F1 par objet selon le reglage, pour les deux ordres de filtre."""
    ev = pd.read_csv(OUT / "kalman_eval_sma.csv")
    ordres = sorted(ev.ordre.unique())
    fig, axes = plt.subplots(1, len(ordres), figsize=(12, 3.8), sharey=True,
                             squeeze=False)
    for ax, ordre in zip(axes[0], ordres):
        d = ev[ev.ordre == ordre]
        noms = [c for c in NOMS if c in set(d.config)]
        bp = ax.boxplot([d[d.config == c].f1.to_numpy() for c in noms],
                        patch_artist=False, widths=0.5,
                        medianprops=dict(color=MODELE, lw=1.4),
                        boxprops=dict(color="black", lw=0.8),
                        whiskerprops=dict(color="black", lw=0.8),
                        capprops=dict(color="black", lw=0.8),
                        flierprops=dict(marker=".", ms=3, mfc="grey", mec="none"))
        for k, c in enumerate(noms):
            micro = agrege(d[d.config == c].to_dict("records"))["f1"]
            ax.plot(k + 1, micro, marker="D", ms=5, color=MANOEUVRE, zorder=5,
                    label="F1 micro-moyennée" if k == 0 else None)
        ax.set_xticks(range(1, len(noms) + 1))
        ax.set_xticklabels([NOMS[c] for c in noms], fontsize=7.5)
        ax.set_title(f"Filtre d'ordre {ordre}", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        grille(ax)
    axes[0][0].set_ylabel("F1 par objet", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "kalman_configs.png")
    plt.close(fig)
    print("kalman_configs.png")


# ------------------------------------------------------------- 4. le tableau

def _micro(d):
    tp, fp, fn = int(d.tp.sum()), int(d.fp.sum()), int(d.fn.sum())
    an = float(d.duree_an.sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r,
            "F1": 2 * p * r / (p + r) if p + r else 0.0,
            "F2": 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0,
            "FA": fp / an if an else np.nan}


def _n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def tables():
    """Un seul tableau : les quatre reglages, sur les 201 objets."""
    ev = pd.read_csv(OUT / "kalman_eval_sma.csv")
    etiq = {"individuel": "params individuels (200 optimisations)",
            "cluster_brutes": "params par cluster, features brutes (8)",
            "cluster_embeddings": "params par cluster, embeddings (5)",
            "moyen": "params moyens (1)"}
    lignes = [r"\begin{tabular}{llrrrccccc}", r"\toprule",
              r"ordre & réglage & TP & FP & FN & précision & rappel & F1 & F2 "
              r"& FA / objet-an \\", r"\midrule"]
    for k, ordre in enumerate(sorted(ev.ordre.unique())):
        for j, (c, nom) in enumerate(etiq.items()):
            m = _micro(ev[(ev.ordre == ordre) & (ev.config == c)])
            tete = str(ordre) if j == 0 else ""
            lignes.append(f"{tete} & {nom} & {m['tp']} & {m['fp']} & {m['fn']} & "
                          f"{_n(m['P'])} & {_n(m['R'])} & {_n(m['F1'])} & "
                          f"{_n(m['F2'])} & {_n(m['FA'], 1)} \\\\")
        lignes.append(r"\midrule" if k == 0 else r"\bottomrule")
    lignes.append(r"\end{tabular}")
    (FIG / "kalman_tab_resultats.tex").write_text("\n".join(lignes) + "\n",
                                                  encoding="utf-8")
    print("  kalman_tab_resultats.tex")


FIGURES = {"nis": fig_nis, "detections": fig_detections, "configs": fig_configs,
           "tables": tables}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=list(FIGURES))
    a = p.parse_args()
    applique()
    FIG.mkdir(parents=True, exist_ok=True)
    for nom in a.only:
        FIGURES[nom]()


if __name__ == "__main__":
    main()
