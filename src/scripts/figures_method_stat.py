"""Figures du rapport sur le test statistique de rupture.

Ecrites dans rapport/method_stat/figures/. Les tableaux sont generes en fragments
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
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maneuver_detection.detecteurs import d2_filtre_adapte, serie, sommets  # noqa: E402
from viz.style_rapport import (  # noqa: E402
    MANOEUVRE, MODELE, SERIE, SEUIL, STAT,
    _axis_label, applique, detections, grille, manoeuvres,
)

FIG = ROOT / "rapport" / "method_stat" / "figures"
OUT = ROOT / "outputs" / "eval_doris200"
TLE = OUT / "tle_doris200.parquet"
LABELS = ROOT / "data/parsed/labelled_leo_DORIS/leo_maneuvers_label_augmented_sso_200.csv"
DEMI = 12
SEUIL_PROD = 20.0
SEUIL_CANAL = {"sma": 20.0, "inclination": 30.0}

SELECTION = [(43609, "sma"), (41727, "sma"), (40336, "sma"), (38257, "sma"),
             (39150, "sma"), (43260, "sma"), (43619, "sma"), (47932, "sma"),
             (41335, "inclination"), (40697, "inclination"),
             (33331, "inclination"), (60989, "inclination")]


def charge(norad):
    df = (pl.scan_parquet(TLE).filter(pl.col("norad") == norad)
          .select("epoch", "sma", "inclination").sort("epoch").collect().to_pandas())
    return serie(df)


def labels():
    lab = pd.read_csv(LABELS)
    lab["epoch"] = pd.to_datetime(lab.epoch, format="mixed", utc=True).dt.tz_localize(None)
    return lab


def _detendu(t, y, fenetre="30D"):
    """Residu a une mediane glissante : une marche de 30 m est invisible sur une
    serie qui parcourt plusieurs kilometres."""
    return y - pd.Series(y, index=pd.to_timedelta(t, "D")).rolling(
        fenetre, center=True, min_periods=12).median().to_numpy()


def _fenetre_lisible(t, vd, largeur, n_min=3, n_max=6):
    """Debut d'une fenetre contenant 3 a 6 manoeuvres, hors bords de serie.

    Ces objets manoeuvrent jusqu'a 49 fois par an : sur l'historique complet la
    figure ne serait qu'une foret de traits verticaux.
    """
    interieur = (t > t[0] + 30) & (t < t[-1] - 30)
    ti = t[interieur]
    debuts = np.arange(ti[0], max(ti[-1] - largeur, ti[0] + 1), largeur / 3)
    n_man = np.array([((vd >= d) & (vd < d + largeur)).sum() for d in debuts])
    ok = np.flatnonzero((n_man >= n_min) & (n_man <= n_max))
    d0 = debuts[ok[len(ok) // 2]] if len(ok) else debuts[int(np.argmax(n_man))]
    return d0, interieur


# ------------------------------------------------------------ 1. le modele

def fig_modele(norad=43437, centre=1499):
    """Une fenetre de 24 TLE et le modele ajuste dessus, sur toute la largeur."""
    t, a, _, ep, _ = charge(norad)
    sl = slice(centre - DEMI + 1, centre + DEMI + 1)
    tau = t[sl] - 0.5 * (t[centre] + t[centre + 1])
    y = a[sl]
    y0 = np.median(y)

    X = np.column_stack([np.ones_like(tau), tau, (tau > 0).astype(float),
                         np.maximum(tau, 0.0)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1, A, B = beta
    g = np.linspace(tau.min(), tau.max(), 600)
    fond = c0 + c1 * g
    modele = fond + A * (g > 0) + B * np.maximum(g, 0.0)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(tau, y - y0, "o", ms=4, color=SERIE, label="TLE")
    ax.plot(g, fond - y0, color=SEUIL, ls="--", lw=1.0,
            label="polynôme seul  $c_0 + c_1\\tau$")
    ax.plot(g, modele - y0, color=MODELE, lw=1.3,
            label="modèle complet  $+\\ A\\,\\mathbb{1}[\\tau>0] + B\\max(\\tau,0)$")
    ax.axvline(0, color="black", lw=0.7, ls=":", alpha=0.8, label="Manoeuvre")
    ax.annotate("", xy=(0.0, A / 2), xytext=(0.0, -A / 2),
                arrowprops=dict(arrowstyle="<->", color=MODELE, lw=0.9))
    ax.text(0.35, 0, f"$A$ = {A:.0f} m", fontsize=9, va="center")
    ax.set_xlabel(r"$\tau$ — jours depuis le milieu de la fenêtre", fontsize=9)
    ax.set_ylabel("Écart au demi-grand axe médian [m]", fontsize=9)
    ax.set_xlim(tau.min(), tau.max())
    ax.legend(loc="center right", fontsize=8)
    grille(ax)
    fig.suptitle(f"NORAD {norad} — fenêtre de {2 * DEMI} TLE", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "stat_modele.png")
    plt.close(fig)
    print(f"stat_modele.png   (norad {norad}, A = {A:.0f} m, B = {B:.1f} m/j)")


def fig_tstat(norad=43437, largeur=360):
    """La statistique de test le long de la serie, sur toute la largeur."""
    t, a, _, ep, _ = charge(norad)
    lab = labels()
    v = lab[(lab.norad_id == norad) & (lab.maneuver_type == "in-track")]
    vd = (v.epoch.to_numpy().astype("datetime64[ns]") - ep[0]).astype(float) / 86400e9
    ts = d2_filtre_adapte(t, a, demi=DEMI, ordre=1, retourne_courbe=True)

    d0, interieur = _fenetre_lisible(t, vd, largeur, n_min=5, n_max=12)
    m = interieur & (t >= d0) & (t <= d0 + largeur)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(t[m] - d0, ts[m], lw=0.8, color=MODELE, label="$|T|$")
    ax.axhline(SEUIL_PROD, color=SEUIL, ls="--", lw=0.8,
               label=f"seuil = {SEUIL_PROD:.0f}")
    manoeuvres(ax, vd[(vd >= d0) & (vd <= d0 + largeur)] - d0)
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, None)
    ax.set_xlabel("Jours", fontsize=9)
    ax.set_ylabel("$|T|$", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    grille(ax)
    fig.suptitle(f"NORAD {norad} — statistique de test", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "stat_tstat.png")
    plt.close(fig)
    print("stat_tstat.png")


# ------------------------------------------------- 2. Student et choix du seuil

def fig_student():
    """Densité et fonction de répartition de la loi de Student, et le seuil.

    Le seuil s est le quantile d'ordre 1 - alpha/2 : la zone de rejet est la queue
    au-dela de s (et son symetrique, le test etant bilateral).
    """
    seuils = pd.read_csv(OUT / "stat_seuils_theoriques.csv")
    s1 = seuils[seuils.ordre == 1]
    nu = int(s1.nu.iloc[0])
    s_theo = float(s1[s1.f_max == 1].seuil_theorique.iloc[0])

    x = np.linspace(-6, 6, 1200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    ax = axes[0]
    ax.plot(x, student.pdf(x, nu), lw=1.4, color=MODELE)
    for signe in (-1, 1):
        q = x[signe * x >= s_theo]
        ax.fill_between(q, 0, student.pdf(q, nu), color=MANOEUVRE, alpha=0.35, lw=0)
    ax.axvline(s_theo, color=MANOEUVRE, lw=0.8)
    ax.axvline(-s_theo, color=MANOEUVRE, lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, None)
    ax.set_xlabel("$t$", fontsize=9)
    ax.set_ylabel("Densité", fontsize=9)
    ax.set_title(rf"Densité de $t_{{{nu}}}$", fontsize=10)
    grille(ax)

    ax = axes[1]
    ax.plot(x, student.cdf(x, nu), lw=1.4, color=MODELE)
    ax.axvline(s_theo, color=MANOEUVRE, lw=0.8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$t$", fontsize=9)
    ax.set_ylabel("$F(t)$", fontsize=9)
    ax.set_title(rf"Fonction de répartition de $t_{{{nu}}}$", fontsize=10)
    grille(ax)

    for ax in axes:
        ax.annotate(rf"$s = {s_theo:.2f}$", (s_theo, ax.get_ylim()[1] * 0.92),
                    fontsize=9, color=MANOEUVRE, ha="left",
                    xytext=(4, 0), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(FIG / "stat_student.png")
    plt.close(fig)
    print(f"stat_student.png   (nu = {nu}, s = {s_theo:.2f})")


# --------------------------------------------------------- 3. les detections

def fig_detections(largeur=130):
    """Douze objets aux profils varies, du plus calme au plus actif."""
    lab = labels()
    types = {"sma": ("in-track", "radial"), "inclination": ("cross-track",)}
    facteur = {"sma": 1.0, "inclination": 1000.0}   # sma deja en m, inclinaison en deg
    unite = {"sma": "Écart au demi-grand axe [m]",
             "inclination": "Écart à l'inclinaison [m°]"}

    n = len(SELECTION)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(12, 1.9 * ((n + 1) // 2)),
                             squeeze=False)
    for ax, (norad, canal) in zip(axes.ravel(), SELECTION):
        t, a, inc, ep, _ = charge(norad)
        y = (a if canal == "sma" else inc) * facteur[canal]
        det = _detendu(t, y)
        ts = d2_filtre_adapte(t, y, demi=DEMI, ordre=1, retourne_courbe=True)
        idx = sommets(ts, SEUIL_CANAL[canal], 8)
        v = lab[(lab.norad_id == norad) & (lab.maneuver_type.isin(types[canal]))]
        vd = (v.epoch.to_numpy().astype("datetime64[ns]") - ep[0]).astype(float) / 86400e9

        d0, interieur = _fenetre_lisible(t, vd, largeur)
        m = interieur & (t >= d0) & (t <= d0 + largeur)

        ax.plot(t[m] - d0, det[m], color=SERIE, lw=1.0)
        manoeuvres(ax, vd[(vd >= d0) & (vd <= d0 + largeur)] - d0)
        detections(ax, [t[k] - d0 for k in idx if m[k]])
        ax.set_title(f"NORAD {norad} — "
                     f"{'in-track' if canal == 'sma' else 'cross-track'}", fontsize=9)
        ax.set_ylabel(unite[canal], fontsize=7.5)
        ax.set_xlim(0, largeur)
        grille(ax)
    for ax in axes.ravel()[-2:]:
        ax.set_xlabel("Jours", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, framealpha=0.7,
               bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout()
    fig.savefig(FIG / "stat_detections_sso.png")
    plt.close(fig)
    print(f"stat_detections_sso.png   ({n} objets)")


# ----------------------------------------------------------- 4. le tableau

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
    """Un seul tableau : les metriques sur les 201 objets, au point de fonctionnement
    optimal de chaque canal (ordre 1 avec terme de pente, seuil maximisant la F1)."""
    sw = pd.read_csv(OUT / "stat_sweep.csv")
    lignes = [r"\begin{tabular}{lcrrrccccc}", r"\toprule",
              r"canal & seuil & TP & FP & FN & précision & rappel & F1 & F2 "
              r"& FA / objet-an \\", r"\midrule"]
    for canal, nom in (("sma", "in-track (demi-grand axe)"),
                       ("inclination", "cross-track (inclinaison)")):
        d = sw[(sw.canal == canal) & (sw.kink) & (sw.ordre == 1)]
        g = d.groupby("seuil").apply(lambda x: pd.Series(_micro(x)), include_groups=False)
        seuil = g.F1.idxmax()
        m = g.loc[seuil]
        lignes.append(f"{nom} & {seuil:.0f} & {int(m.tp)} & {int(m.fp)} & {int(m.fn)} & "
                      f"{_n(m.P)} & {_n(m.R)} & {_n(m.F1)} & {_n(m.F2)} & "
                      f"{_n(m.FA, 2)} \\\\")
    lignes += [r"\bottomrule", r"\end{tabular}"]
    (FIG / "stat_tab_resultats.tex").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print("  stat_tab_resultats.tex")


FIGURES = {"modele": fig_modele, "tstat": fig_tstat, "student": fig_student,
           "detections": fig_detections, "tables": tables}


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
