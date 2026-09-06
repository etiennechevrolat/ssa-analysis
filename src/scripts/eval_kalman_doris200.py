"""Evaluation du filtre de Kalman sur le jeu DORIS augmente (201 objets SSO).

Trois etages, a lancer dans l'ordre :

  --stage opt      optimisation (var_Q, r, p0) objet par objet  [long, ~1 h sur 15 coeurs]
  --stage cluster  clustering des objets + un jeu de parametres par cluster
  --stage eval     les 4 configurations mises en concurrence, metriques + tableaux LaTeX

Pourquoi un script et pas optimise_seuil_kalman_via_regul_gaussienne tel quel :

  - minimize_DE (src/optimisation_seuil/metrics.py:58) fige popsize=20, maxiter=200,
    soit ~12 000 evaluations de la loss. A 40 us par point TLE et 838 689 points, cela
    represente ~12 h CPU par ordre. Le budget est reduit ici, et on enregistre
    `converge` (l'optimiseur s'est-il arrete avant maxiter) pour verifier que la
    reduction ne coupe pas l'optimisation en plein vol ;
  - la branche man_ilrs=False de optimise_seuil.py ne restreint PAS les manoeuvres a la
    couverture temporelle des TLE : toute manoeuvre hors couverture compte en FN et
    plafonne le rappel independamment du detecteur. On restreint ici ;
  - l'appariement final passe par eval_common.apparie (1-1 glouton), pas par
    confusion_matrix (non exclusif), pour etre comparable au test statistique.

La loss, les bornes, alpha et le seuil chi2 sont identiques a l'original.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from maneuver_detection.discrete_kalman_filter import (  # noqa: E402
    kalman_filter_ordre_1,
    kalman_filter_ordre_2,
)
from maneuver_detection.eval_common import (  # noqa: E402
    apparie,
    agrege,
    metriques,
    restreint_fenetre,
    to_days,
)
from optimisation_seuil.metrics import lissage_noyau_gaussien_metriques  # noqa: E402

TLE = ROOT / "outputs/eval_doris200/tle_doris200.parquet"
LABELS = ROOT / "data/parsed/labelled_leo_DORIS/leo_maneuvers_label_augmented_sso_200.csv"
OUT = ROOT / "outputs/eval_doris200"

ALPHA = 0.997
SEUIL_CHI2 = chi2.ppf(ALPHA, df=1)
SIGMA_LISSAGE, BETA_LISSAGE = 1.0, 0.5
TOL_DEDUP_J = 1.0        # deux detections a moins de 1 j = la meme manoeuvre
BORNES = [(-10.0, 1.0), (-10.0, 2.0), (0.0, 5.0)]   # log10 de var_Q, r, p0
BSTAR_FACTOR_K = 1.0     # cf. configs/ml/task/finetuning.yaml:15

## Le canal porte le type de manoeuvre qu'il peut voir : une manoeuvre in-track change
## le demi-grand axe, une manoeuvre cross-track change l'inclinaison. Evaluer le canal
## sma contre les manoeuvres cross-track ajouterait des FN que le canal ne peut pas voir.
TYPES_CANAL = {"sma": ("in-track", "radial"), "inclination": ("cross-track",)}


# ----------------------------------------------------------------- donnees

def charge_labels(canal, source=None):
    lab = pd.read_csv(LABELS)
    lab["epoch"] = pd.to_datetime(lab.epoch, format="mixed", utc=True).dt.tz_localize(None)
    lab = lab[lab.maneuver_type.isin(TYPES_CANAL[canal])]
    if source:
        lab = lab[lab.source == source]
    return lab


def serie(norad, canal):
    return (pl.scan_parquet(TLE)
            .filter(pl.col("norad") == norad)
            .select("epoch", canal)
            .sort("epoch")
            .collect())


def nis_de(df, ordre, var_Q, r, p0, canal):
    f = kalman_filter_ordre_1 if ordre == 1 else kalman_filter_ordre_2
    _, nis, _ = f(df, var_Q=var_Q, r=r, p0=p0, metrique=canal)
    return np.asarray(nis, dtype=float)


def detections(epoch_d, nis, seuil=SEUIL_CHI2, tol=TOL_DEDUP_J):
    """Points au-dessus du seuil, dedoublonnes temporellement."""
    pred = epoch_d[nis > seuil]
    if len(pred) == 0:
        return pred
    garde = [pred[0]]
    for t in pred[1:]:
        if t - garde[-1] > tol:
            garde.append(t)
    return np.asarray(garde)


def score(epoch_d, nis, vrai_d, duree_an):
    tp, fp, fn = apparie(detections(epoch_d, nis), vrai_d)
    return {**metriques(tp, fp, fn, duree_an), "duree_an": duree_an}


# ----------------------------------------------------------------- etage opt

def prepare(norad, canal, lab):
    """(df, epoch_d, vrai_d, duree_an) ou None si l'objet est inexploitable."""
    df = serie(norad, canal)
    if df.height < 10:
        return None
    e = df["epoch"].to_numpy()
    epoch = e[1:]                       # la NIS demarre au pas k = 1
    t0 = epoch[0]
    epoch_d = to_days(epoch, t0)
    vrai = lab[lab.norad_id == norad]["epoch"].to_numpy()
    vrai_d = restreint_fenetre(to_days(vrai, t0), epoch_d)
    return df, epoch_d, vrai_d, (epoch_d[-1] - epoch_d[0]) / 365.25


def optimise_un(norad, ordre, canal, lab, maxiter, popsize, seed=0):
    prep = prepare(norad, canal, lab)
    if prep is None:
        return None
    df, epoch_d, vrai_d, duree_an = prep
    if len(vrai_d) == 0:
        return None

    def loss(x):
        try:
            nis = nis_de(df, ordre, 10 ** x[0], 10 ** x[1], 10 ** x[2], canal)
        except Exception:
            return 1.0
        if not np.all(np.isfinite(nis)):
            return 1.0
        return lissage_noyau_gaussien_metriques(
            vrai_d, epoch_d, nis, SEUIL_CHI2, SIGMA_LISSAGE, BETA_LISSAGE)

    t0 = time.perf_counter()
    res = differential_evolution(
        loss, BORNES,
        x0=np.array([np.log10(0.05), np.log10(0.5), np.log10(1000.0)]),
        init="sobol", popsize=popsize, maxiter=maxiter, tol=0.01,
        mutation=(0.5, 1.0), recombination=0.7, polish=False, seed=seed, workers=1)
    var_Q, r, p0 = 10 ** res.x

    nis = nis_de(df, ordre, var_Q, r, p0, canal)
    return {"norad": int(norad), "ordre": ordre, "canal": canal,
            "var_Q": var_Q, "r": r, "p0": p0,
            "n_tle": df.height, "n_vrai": len(vrai_d),
            "loss": float(res.fun), "nit": int(res.nit),
            "converge": bool(res.nit < maxiter), "secondes": time.perf_counter() - t0,
            **score(epoch_d, nis, vrai_d, duree_an)}


def stage_opt(args):
    lab = charge_labels(args.canal)
    norads = sorted(pl.scan_parquet(TLE).select("norad").unique().collect()["norad"].to_list())
    if args.limite:
        norads = norads[: args.limite]
    taches = [(n, o) for o in args.ordres for n in norads]
    print(f"[opt] {len(taches)} taches ({len(norads)} objets x {len(args.ordres)} ordres), "
          f"canal={args.canal}, maxiter={args.maxiter}, popsize={args.popsize}, "
          f"n_jobs={args.n_jobs}", flush=True)

    t0 = time.perf_counter()
    res = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(optimise_un)(n, o, args.canal, lab, args.maxiter, args.popsize)
        for n, o in taches)
    res = [r for r in res if r]

    df = pd.DataFrame(res)
    out = OUT / f"kalman_opt_{args.canal}.csv"
    df.to_csv(out, index=False)
    print(f"[opt] {len(df)} lignes -> {out}  ({(time.perf_counter()-t0)/60:.1f} min)")
    print(f"[opt] converge avant maxiter : {df.converge.mean():.0%}")
    for o in sorted(df.ordre.unique()):
        s = df[df.ordre == o]
        print(f"[opt] ordre {o} : F1 median {s.f1.median():.3f}, "
              f"micro {agrege(s.to_dict('records'))['f1']:.3f}")


# --------------------------------------------------------- etage invariance

def stage_invariance(args):
    """Combien de parametres le filtre a-t-il REELLEMENT ?

    Deux verifications numeriques, sur des objets reels :

      1. (var_Q, r, p0) -> (c var_Q, c r, c p0) doit laisser le gain de Kalman inchange
         et diviser la NIS par c exactement. Si c'est vrai, le triplet ne porte que DEUX
         informations : le rapport var_Q/r, qui fixe la bande passante du filtre, et une
         echelle qui joue exactement le role du seuil ;
      2. p0 seul, a var_Q et r fixes. Si la detection n'en depend pas, c'est une
         direction morte de l'espace de recherche.

    L'enjeu est concret : un clustering des triplets optimises melange alors une
    dimension informative, une dimension redondante avec le seuil, et une dimension de
    bruit pur -- ce qui suffit a le rendre non significatif.
    """
    from maneuver_detection.detecteurs import bruit_hf

    lignes = []
    for norad in args.norads:
        df = serie(norad, args.canal)
        y = df[args.canal].to_numpy()
        sigma = float(bruit_hf(y))
        base = dict(var_Q=10 * sigma ** 2, r=sigma ** 2, p0=1000.0)
        nis0 = nis_de(df, 1, base["var_Q"], base["r"], base["p0"], args.canal)

        ref = set(np.flatnonzero(nis0 > SEUIL_CHI2))
        for c in (1e-2, 1e-1, 1e1, 1e2):
            nis = nis_de(df, 1, base["var_Q"] * c, base["r"] * c, base["p0"] * c, args.canal)
            ecart = float(np.nanmax(np.abs(nis * c - nis0) / np.maximum(np.abs(nis0), 1e-12)))
            ## le critere qui compte n'est pas l'ecart numerique brut -- il accumule
            ## l'arrondi sur des dizaines de milliers de pas de recurrence -- mais le
            ## fait que le seuil divise par c redonne EXACTEMENT les memes detections.
            d = set(np.flatnonzero(nis > SEUIL_CHI2 / c))
            lignes.append({"test": "echelle", "norad": int(norad), "c": c,
                           "ecart_relatif_max": ecart,
                           "jaccard_detections": len(d & ref) / max(len(d | ref), 1)})
        for p0 in (1e-3, 1e0, 1e5):
            nis = nis_de(df, 1, base["var_Q"], base["r"], p0, args.canal)
            d = set(np.flatnonzero(nis > SEUIL_CHI2))
            lignes.append({"test": "p0", "norad": int(norad), "p0": p0,
                           "n_det": len(d), "n_det_ref": len(ref),
                           "jaccard": len(d & ref) / max(len(d | ref), 1)})
        print(f"[inv] norad {norad} : bruit HF {sigma*1000:.2f} m, {len(ref)} détections "
              f"de référence", flush=True)

    df = pd.DataFrame(lignes)
    df.to_csv(OUT / "kalman_invariance.csv", index=False)
    e = df[df.test == "echelle"].ecart_relatif_max.max()
    je = df[df.test == "echelle"].jaccard_detections.min()
    jp = df[df.test == "p0"].jaccard.min()
    print(f"[inv] échelle : écart relatif max {e:.1e}, Jaccard des détections min {je:.4f}")
    print(f"[inv] p0 sur 8 décades : Jaccard min {jp:.4f}")
    print(f"[inv] -> {'2' if je > 0.99 and jp > 0.99 else '3'} paramètres effectifs "
          f"sur les 3 optimisés")


# ------------------------------------------------------------- etage cluster

def features_brutes():
    """Grandeurs disponibles sans rien optimiser : geometrie de l'orbite, cadence du
    catalogue, niveau de bruit de la serie."""
    from maneuver_detection.detecteurs import bruit_hf

    df = pl.read_parquet(TLE).to_pandas()
    lignes = []
    for norad, g in df.groupby("norad"):
        e = g["epoch"].to_numpy()
        dt = np.diff(e) / np.timedelta64(1, "D")
        lignes.append({"norad": int(norad),
                       "sma": float(g.sma.median()),
                       "inclination": float(g.inclination.median()),
                       "eccentricity": float(g.eccentricity.median()),
                       "bstar": float(np.arcsinh(g.bstar.median() / 1e-4)),
                       "cadence_j": float(np.median(dt)) if len(dt) else np.nan,
                       "bruit_hf": float(np.log10(bruit_hf(g.sma.to_numpy()) + 1e-12))})
    return pd.DataFrame(lignes).set_index("norad")


def features_embeddings(ckpt):
    """Representation 128-D par objet, issue de l'encodeur preentraine (token CLS moyenne)."""
    import torch
    from ml.datahandler import build_features
    from ml.inference import embed_objects, load_pretrained_backbone
    from sklearn.preprocessing import StandardScaler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, cfg, mean, scale = load_pretrained_backbone(ckpt, device)
    print(f"[cluster] backbone {cfg.model.name} embed_dim={cfg.model.encoder_embed_dim} "
          f"window={cfg.data.window_size} stride={cfg.data.stride} device={device}")

    ## load_doris_objects fait exactement le pretraitement du pretrain : deduplication
    ## d'epoch, retrait des outliers sma, dt = log1p(diff), bstar = asinh(bstar/facteur).
    ## Le reimplementer ici donnerait des features silencieusement decalees de celles
    ## que l'encodeur a vues a l'entrainement.
    from ml.datahandler import load_doris_objects
    objets, _ = load_doris_objects(
        ROOT / "data/parsed/labelled_leo_DORIS", BSTAR_FACTOR_K,
        labels_file="leo_maneuvers_label_augmented_sso_200.csv",
        params_file="leo_doris_orbital_params_sso.parquet")

    per_obj = {}
    for norad, g in objets.items():
        feats, cols = build_features(g.reset_index(drop=True), spacetrack=True)
        X = feats[cols].to_numpy(np.float32)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < cfg.data.window_size:
            continue
        if mean is None:
            sc = StandardScaler().fit(X)
            X = (X - sc.mean_.astype(np.float32)) / sc.scale_.astype(np.float32)
        else:
            X = (X - mean) / scale
        per_obj[int(norad)] = X
    print(f"[cluster] {len(per_obj)} objets assez longs pour une fenetre de "
          f"{cfg.data.window_size}")
    emb = embed_objects(backbone, per_obj, device,
                        window_size=cfg.data.window_size, stride=cfg.data.stride)
    return pd.DataFrame(emb).T.sort_index()


def hdbscan_ajuste(X, n_pca=10):
    """HDBSCAN reglé pour du TRANSFERT de parametres, pas pour un score de clustering.

    Deux ecueils rencontres en reglant naivement sur la silhouette :

      - la silhouette recompense un decoupage degenere. Sur les features brutes, son
        maximum (0.75) est un partage 195 / 5 : le cluster de 195 objets redonne
        exactement la configuration "parametres moyens", le transfert n'apporte rien ;
      - HDBSCAN sur les 128 dimensions brutes de l'embedding renvoie 48 % de bruit
        (201 points dans 128 dimensions). On reduit donc par ACP a 10 composantes,
        qui portent 79 % de la variance, avant de clusteriser.

    D'ou une selection SOUS CONTRAINTE : entre 3 et 12 clusters d'au moins 6 membres,
    au plus 35 % de bruit, aucun cluster au-dela de 75 % des objets. La silhouette ne
    sert qu'a departager les configurations qui satisfont deja ces conditions.
    Le plafond de 12 est budgetaire : chaque cluster demande sa propre optimisation.
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    if Xs.shape[1] > 20:
        Xs = PCA(n_pca, random_state=0).fit_transform(Xs)

    best = None
    for mcs in (6, 8, 10, 12, 15, 20):
        for ms in (1, 3, 5):
            lab = HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms),
                          copy=True).fit_predict(Xs)
            n_cl = len(set(lab) - {-1})
            bruit = float((lab == -1).mean())
            m = lab != -1
            if not (3 <= n_cl <= 12) or bruit > 0.35 or m.sum() <= n_cl:
                continue
            part_max = float(pd.Series(lab[m]).value_counts().max() / len(lab))
            if part_max > 0.75:
                continue
            s = float(silhouette_score(Xs[m], lab[m]))
            if best is None or s > best[0]:
                best = (s, mcs, ms, lab, n_cl, bruit, part_max)
    if best is None:
        return np.full(len(Xs), -1), {"min_cluster_size": None, "n_clusters": 0}
    s, mcs, ms, lab, n_cl, bruit, part_max = best
    return lab, {"min_cluster_size": int(mcs), "min_samples": int(ms),
                 "n_clusters": n_cl, "bruit": round(bruit, 3),
                 "part_plus_gros": round(part_max, 3), "silhouette": round(s, 3)}


def optimise_cluster(norads, ordre, canal, lab, maxiter, popsize, n_sous=6):
    """Un seul jeu (var_Q, r, p0) pour tout un cluster.

    La loss est la moyenne des loss lissees d'un sous-echantillon : on ne peut pas
    sommer des F1 par objet sans que les objets les plus longs ecrasent les autres, et
    on ne peut pas non plus optimiser sur tout le cluster sans exploser le budget.
    Le sous-echantillon prend les objets de longueur mediane, pas les extremes.
    """
    preps = []
    for n in norads:
        p = prepare(n, canal, lab)
        if p and len(p[2]) > 0:
            preps.append((n, p))
    if not preps:
        return None
    preps.sort(key=lambda x: x[1][0].height)
    milieu = len(preps) // 2
    demi = n_sous // 2
    choisis = preps[max(0, milieu - demi): max(0, milieu - demi) + n_sous]

    def loss(x):
        tot = 0.0
        for _, (df, epoch_d, vrai_d, _) in choisis:
            try:
                nis = nis_de(df, ordre, 10 ** x[0], 10 ** x[1], 10 ** x[2], canal)
            except Exception:
                return 1.0
            if not np.all(np.isfinite(nis)):
                return 1.0
            tot += lissage_noyau_gaussien_metriques(
                vrai_d, epoch_d, nis, SEUIL_CHI2, SIGMA_LISSAGE, BETA_LISSAGE)
        return tot / len(choisis)

    res = differential_evolution(
        loss, BORNES, x0=np.array([np.log10(0.05), np.log10(0.5), np.log10(1000.0)]),
        init="sobol", popsize=popsize, maxiter=maxiter, tol=0.01,
        mutation=(0.5, 1.0), recombination=0.7, polish=False, seed=0, workers=1)
    var_Q, r, p0 = 10 ** res.x
    return {"var_Q": var_Q, "r": r, "p0": p0, "n_membres": len(norads),
            "n_optimises": len(choisis), "loss": float(res.fun)}


def stage_cluster(args):
    lab = charge_labels(args.canal)
    opt = pd.read_csv(OUT / f"kalman_opt_{args.canal}.csv")
    norads = sorted(opt.norad.unique())

    jeux = {}
    fb = features_brutes()
    jeux["brutes"] = fb.loc[[n for n in norads if n in fb.index]]
    if args.embeddings:
        emb = features_embeddings(Path(args.ckpt))
        jeux["embeddings"] = emb.loc[[n for n in norads if n in emb.index]]

    appartenance, taches = [], []
    for nom, X in jeux.items():
        labels, info = hdbscan_ajuste(X)
        print(f"[cluster] {nom:11s} {len(X)} objets -> {info}", flush=True)
        appartenance += [{"famille": nom, "norad": int(n), "cluster": int(c)}
                         for n, c in zip(X.index, labels)]
        X.to_csv(OUT / f"kalman_features_{nom}.csv")
        for ordre in args.ordres:
            for c in sorted(set(labels) - {-1}):
                membres = [int(n) for n, l in zip(X.index, labels) if l == c]
                taches.append((nom, int(c), ordre, membres))

    ## Une optimisation par cluster coute n_sous fois une optimisation d'objet : en
    ## sequentiel les ~26 taches demanderaient plusieurs heures. On parallelise sur les
    ## taches, chacune restant mono-thread (workers=1 dans differential_evolution).
    def _une(nom, c, ordre, membres):
        t0 = time.perf_counter()
        p = optimise_cluster(membres, ordre, args.canal, lab,
                             args.maxiter, args.popsize, args.n_sous)
        if not p:
            return None
        print(f"[cluster] {nom} c={c} ordre={ordre} var_Q={p['var_Q']:.3g} "
              f"r={p['r']:.3g} ({p['n_membres']} membres, "
              f"{time.perf_counter()-t0:.0f} s)", flush=True)
        return {"famille": nom, "cluster": c, "ordre": ordre,
                "secondes": time.perf_counter() - t0, **p}

    print(f"[cluster] {len(taches)} optimisations de cluster, n_jobs={args.n_jobs}",
          flush=True)
    resume = [r for r in Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_une)(*t) for t in taches) if r]
    pd.DataFrame(appartenance).to_csv(OUT / f"kalman_clusters_{args.canal}.csv", index=False)
    pd.DataFrame(resume).to_csv(OUT / f"kalman_params_clusters_{args.canal}.csv", index=False)
    print(f"[cluster] -> kalman_clusters_{args.canal}.csv, "
          f"kalman_params_clusters_{args.canal}.csv")


# ---------------------------------------------------------------- etage eval

def evalue_config(norad, ordre, canal, lab, params):
    prep = prepare(norad, canal, lab)
    if prep is None:
        return None
    df, epoch_d, vrai_d, duree_an = prep
    if len(vrai_d) == 0:
        return None
    try:
        nis = nis_de(df, ordre, params["var_Q"], params["r"], params["p0"], canal)
    except Exception:
        return None
    if not np.all(np.isfinite(nis)):
        return None
    return {"norad": int(norad), **score(epoch_d, nis, vrai_d, duree_an)}


def stage_eval(args):
    lab = charge_labels(args.canal)
    opt = pd.read_csv(OUT / f"kalman_opt_{args.canal}.csv")
    clusters = pd.read_csv(OUT / f"kalman_clusters_{args.canal}.csv")
    pcl = pd.read_csv(OUT / f"kalman_params_clusters_{args.canal}.csv")

    lignes = []
    for ordre in sorted(opt.ordre.unique()):
        o = opt[opt.ordre == ordre].set_index("norad")
        moyens = {k: float(np.exp(np.log(o[k]).mean())) for k in ("var_Q", "r", "p0")}

        configs = {"individuel": lambda n: o.loc[n, ["var_Q", "r", "p0"]].to_dict()
                   if n in o.index else None,
                   "moyen": lambda n: moyens}
        for fam in clusters.famille.unique():
            app = clusters[clusters.famille == fam].set_index("norad").cluster
            tbl = pcl[(pcl.famille == fam) & (pcl.ordre == ordre)].set_index("cluster")

            def f(n, app=app, tbl=tbl):
                c = app.get(n, -1)
                if c in tbl.index:
                    return tbl.loc[c, ["var_Q", "r", "p0"]].to_dict()
                return moyens          # repli sur les params moyens pour le bruit
            configs[f"cluster_{fam}"] = f

        for nom, f in configs.items():
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(evalue_config)(int(n), ordre, args.canal, lab, f(int(n)))
                for n in o.index if f(int(n)) is not None)
            res = [r for r in res if r]
            for r in res:
                lignes.append({"config": nom, "ordre": ordre, **r})
            print(f"[eval] ordre {ordre} {nom:22s} {len(res):3d} objets  "
                  f"F1 micro {agrege(res)['f1']:.3f}", flush=True)
    df = pd.DataFrame(lignes)
    df.to_csv(OUT / f"kalman_eval_{args.canal}.csv", index=False)
    print(f"[eval] {len(df)} lignes -> kalman_eval_{args.canal}.csv")


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["opt", "cluster", "eval", "invariance"])
    p.add_argument("--embeddings", action="store_true")
    p.add_argument("--ckpt", default=str(ROOT / "outputs/ml/pretrain/2026-09-02_16-13-15/checkpoints/best.pt"))
    p.add_argument("--n-sous", type=int, default=6)
    p.add_argument("--norads", type=int, nargs="+",
                   default=[43437, 41335, 36508, 27386, 39086])
    p.add_argument("--canal", default="sma", choices=["sma", "inclination"])
    p.add_argument("--ordres", type=int, nargs="+", default=[1, 2])
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--n-jobs", type=int, default=15)
    p.add_argument("--limite", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"opt": stage_opt, "cluster": stage_cluster, "eval": stage_eval,
     "invariance": stage_invariance}[args.stage](args)


if __name__ == "__main__":
    main()
