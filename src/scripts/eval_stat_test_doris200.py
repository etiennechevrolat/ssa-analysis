"""Evaluation du test statistique de rupture sur le jeu DORIS augmente (201 objets SSO).

Trois etages :

  --stage sweep   ordre du polynome x terme de pente x seuil, sur les deux canaux
  --stage cond    conditionnement de X^T X en fonction de l'ordre et de la demi-fenetre
  --stage seuil   table de seuils de Student + loi empirique de |T| sur arcs propres

Ce que le detecteur fait (maneuver_detection/detecteurs.py, d2_filtre_adapte) :
sur chaque fenetre de N = 2*demi points recentree sur le milieu de l'intervalle, on ajuste
en une seule regression

    y = c_0 + c_1 tau + ... + c_p tau^p + A 1[tau > 0] + B max(tau, 0)

et le score est la t-statistique de A. Le seuil de production vaut 20.0, en dur, et n'a
aucun lien avec le quantile de Student de method_stat.tex : cet ecart est precisement ce
que l'etage `seuil` mesure.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maneuver_detection.detecteurs import d2_filtre_adapte, pas_quantification, serie, sommets  # noqa: E402
from maneuver_detection.eval_common import apparie, metriques, to_days  # noqa: E402

TLE = ROOT / "outputs/eval_doris200/tle_doris200.parquet"
LABELS = ROOT / "data/parsed/labelled_leo_DORIS/leo_maneuvers_label_augmented_sso_200.csv"
CORPUS = ROOT / "sso_annotation/cache/corpus.csv"
SERIES_CORPUS = ROOT / "sso_annotation/cache/series_corpus.parquet"
OUT = ROOT / "outputs/eval_doris200"

DEMI = 12                 # N = 24 points par fenetre
SEPARATION = 8
SEUIL_PROD = 20.0
SEUILS = [3, 4, 5, 6, 8, 10, 14, 20, 30, 45, 70, 110, 170, 260]
ORDRES = [1, 2, 3, 4, 5, 6]
CANAUX = {"sma": ("in-track", "radial"), "inclination": ("cross-track",)}


def charge_labels(source=None):
    """source='DORIS-IDS' isole les labels INDEPENDANTS du detecteur evalue.

    Les deux points de vue sont necessaires et aucun n'est neutre : sur le jeu complet
    la majorite des labels vient du detecteur lui-meme, donc la precision est flattee ;
    sur DORIS seul la verite est incomplete, donc la precision est un MINORANT -- une
    manoeuvre reelle non enregistree par DORIS compte en faux positif.
    """
    lab = pd.read_csv(LABELS)
    lab["epoch"] = pd.to_datetime(lab.epoch, format="mixed", utc=True).dt.tz_localize(None)
    if source:
        lab = lab[lab.source == source]
    return lab


def serie_objet(norad):
    """(t_jours, sma_m, inclinaison_deg, epochs) apres nettoyage, comme en production."""
    df = (pl.scan_parquet(TLE).filter(pl.col("norad") == norad)
          .select("epoch", "sma", "inclination").sort("epoch").collect().to_pandas())
    if len(df) < 4 * DEMI:
        return None
    return serie(df)


# ------------------------------------------------------------------ etage sweep

def sweep_objet(norad, lab, ordres, kinks):
    prep = serie_objet(norad)
    if prep is None:
        return []
    t, a, inc, ep, _ = prep
    t0 = ep[0]
    lignes = []
    for canal, valeurs in (("sma", a), ("inclination", inc)):
        vrai = lab[(lab.norad_id == norad) & (lab.maneuver_type.isin(CANAUX[canal]))]
        vrai_d = to_days(vrai["epoch"].to_numpy(), t0)
        vrai_d = vrai_d[(vrai_d >= t[0]) & (vrai_d <= t[-1])]
        duree_an = (t[-1] - t[0]) / 365.25
        for ordre in ordres:
            for kink in kinks:
                tstat = d2_filtre_adapte(t, valeurs, demi=DEMI, ordre=ordre,
                                         avec_kink=kink, retourne_courbe=True)
                for s in SEUILS:
                    idx = sommets(tstat, s, SEPARATION)
                    tp, fp, fn = apparie(t[idx], vrai_d)
                    lignes.append({"norad": int(norad), "canal": canal, "ordre": ordre,
                                   "kink": kink, "seuil": s, "duree_an": duree_an,
                                   "n_vrai": len(vrai_d), **metriques(tp, fp, fn)})
    return lignes


def stage_sweep(args):
    lab = charge_labels(args.source)
    norads = sorted(pl.scan_parquet(TLE).select("norad").unique().collect()["norad"].to_list())
    if args.source:
        norads = [n for n in norads if n in set(lab.norad_id)]
    if args.limite:
        norads = norads[: args.limite]
    kinks = [True, False] if args.sans_kink else [True]
    print(f"[sweep] {len(norads)} objets x {len(args.ordres)} ordres x {len(kinks)} kink "
          f"x {len(SEUILS)} seuils x 2 canaux, n_jobs={args.n_jobs}", flush=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(sweep_objet)(n, lab, args.ordres, kinks) for n in norads)
    df = pd.DataFrame([l for sub in res for l in sub])
    suffixe = "_doris" if args.source else ""
    out = OUT / f"stat_sweep{suffixe}.csv"
    df.to_csv(out, index=False)
    print(f"[sweep] {len(df)} lignes -> {out}  ({(time.perf_counter()-t0)/60:.1f} min)")


# ------------------------------------------------------------------- etage cond

def stage_cond(args):
    """Conditionnement du plan d'experience en fonction de l'ordre et de la demi-fenetre.

    On mesure aussi le rapport entre la variance de A donnee par la pseudo-inverse et
    celle donnee par l'inverse exacte : c'est ce rapport, et pas le conditionnement seul,
    qui explique la derive du taux de fausse alarme. Quand la matrice decroche, pinv
    tronque les valeurs singulieres et SOUS-estime v_A, donc SUR-estime T.
    """
    lignes = []
    for demi in (6, 8, 12, 16, 24, 36):
        tau = np.arange(-demi + 1, demi + 1, dtype=float) - 0.5
        for ordre in range(1, 9):
            for kink in (True, False):
                cols = [np.ones_like(tau)] + [tau ** k for k in range(1, ordre + 1)]
                cols.append((tau > 0).astype(float))
                if kink:
                    cols.append(np.maximum(tau, 0.0))
                X = np.column_stack(cols)
                q = X.shape[1]
                if len(tau) - q <= 1:
                    continue
                G = X.T @ X
                j = ordre + 1
                v_pinv = np.linalg.pinv(G)[j, j]
                try:
                    v_exact = np.linalg.inv(G)[j, j]
                except np.linalg.LinAlgError:
                    v_exact = np.nan
                lignes.append({"demi": demi, "N": len(tau), "ordre": ordre, "kink": kink,
                               "q": q, "nu": len(tau) - q,
                               "cond": np.linalg.cond(G),
                               "v_A_pinv": v_pinv, "v_A_exact": v_exact,
                               "ratio": v_pinv / v_exact if v_exact else np.nan})
    df = pd.DataFrame(lignes)
    out = OUT / "stat_conditionnement.csv"
    df.to_csv(out, index=False)
    print(f"[cond] {len(df)} lignes -> {out}")
    print(df[(df.demi == 12) & (df.kink)][["ordre", "q", "nu", "cond", "ratio"]].to_string(index=False))


# ------------------------------------------------------------------ etage seuil

def stage_seuil(args):
    """Seuil theorique de Student vs seuil empirique, et loi de |T| sur arcs propres."""
    # 1. table des seuils pour la configuration REELLE du code (N = 24, pas 20)
    lab = charge_labels()
    norads = sorted(pl.scan_parquet(TLE).select("norad").unique().collect()["norad"].to_list())
    duree = (pl.scan_parquet(TLE).group_by("norad")
             .agg((pl.col("epoch").max() - pl.col("epoch").min()).alias("d")).collect())
    an_total = float(duree["d"].dt.total_days().sum()) / 365.25
    n_tle = pl.scan_parquet(TLE).select(pl.len()).collect().item()
    # un test par TLE et par canal
    m = n_tle * len(CANAUX) / an_total
    print(f"[seuil] {n_tle} TLE, {an_total:.0f} objet-an cumules, "
          f"m = {m:.0f} tests par an (2 canaux, pas de 1 TLE)")

    lignes = []
    for ordre in ORDRES:
        nu = 2 * DEMI - (ordre + 3)
        for fmax in (10, 5, 2, 1, 0.5, 0.1):
            alpha = fmax / m
            lignes.append({"ordre": ordre, "nu": nu, "f_max": fmax, "alpha": alpha,
                           "seuil_theorique": student.ppf(1 - alpha / 2, nu)})
    df = pd.DataFrame(lignes)
    df.to_csv(OUT / "stat_seuils_theoriques.csv", index=False)
    print(df[df.ordre == 1].to_string(index=False))

    # 2. loi empirique de |T| sur les temoins passives du corpus SSO
    if SERIES_CORPUS.exists() and CORPUS.exists():
        corpus = pd.read_csv(CORPUS)
        temoins = corpus[corpus.role.astype(str).str.startswith("controle")]
        ids = temoins.norad.tolist()[: args.n_temoins]
        print(f"[seuil] {len(ids)} temoins passives")
        series = pl.scan_parquet(SERIES_CORPUS).filter(pl.col("norad").is_in(ids)).collect()
        vals = {}
        for norad, g in series.to_pandas().groupby("norad"):
            prep = serie(g)
            if prep is None or len(prep[0]) < 4 * DEMI:
                continue
            t, a, inc, ep, _ = prep
            for ordre in (1, 2, 3):
                ts = d2_filtre_adapte(t, a, demi=DEMI, ordre=ordre, retourne_courbe=True)
                vals.setdefault(ordre, []).append(ts[ts > 0])
        emp = {o: np.concatenate(v) for o, v in vals.items() if v}
        np.savez(OUT / "stat_T_empirique.npz", **{f"ordre{o}": v for o, v in emp.items()})
        for o, v in emp.items():
            nu = 2 * DEMI - (o + 3)
            q999 = np.quantile(v, 0.999)
            print(f"[seuil] ordre {o} : n={len(v)}  quantile 99.9% empirique {q999:.2f} "
                  f"vs Student {student.ppf(1 - 0.001 / 2, nu):.2f}")
    else:
        print("[seuil] corpus SSO absent, loi empirique sautee")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["sweep", "cond", "seuil"])
    p.add_argument("--ordres", type=int, nargs="+", default=ORDRES)
    p.add_argument("--sans-kink", action="store_true", help="ajoute la variante sans terme B")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--n-temoins", type=int, default=30)
    p.add_argument("--source", default=None,
                   help="ne garder que les labels de cette source, ex. DORIS-IDS")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"sweep": stage_sweep, "cond": stage_cond, "seuil": stage_seuil}[args.stage](args)


if __name__ == "__main__":
    main()
