"""Évaluation comparative des préentraînements MAE + génération des figures du rapport.

Usage :
    PYTHONPATH=src .venv/bin/python src/scripts/eval_pretrain.py
    PYTHONPATH=src .venv/bin/python src/scripts/eval_pretrain.py --only channels,km

Le problème que ce script résout : aucune val loss n'est comparable d'un run à l'autre. Le jeu
de features a changé le 18/08 (11 -> 12 canaux), les poids par canal le 25/08, la normalisation
le 26/08 ; s'y ajoutent des fenêtres et des taux de masquage différents, qui changent la
difficulté de la tâche elle-même. On ré-évalue donc tous les checkpoints à 12 canaux sous un
protocole commun, en unités physiques, et on pousse jusqu'à la représentation elle-même.

Les checkpoints à 11 canaux (avant le 18/08) ne sont PAS ré-évaluables : le constructeur de
features qui les a produits n'existe plus. Ils n'apparaissent que par leurs courbes loggées.

Modules (option --only) :
  curves    courbes d'entraînement de tous les runs
  channels  décomposition MSE / variance / skill par canal
  km        erreur sur sma_local en kilomètres, par amplitude du patch masqué
  mask      matrice de transfert modèle x taux de masquage d'évaluation
  steps     restitution des ruptures de niveau (masquage forcé du patch)
  embed     clustering HDBSCAN et sonde linéaire sur les embeddings CLS
  examples  reconstructions qualitatives

Figures dans rapport/figures/ (préfixe pretrain_), métriques dans pretrain_metrics.json.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, silhouette_score
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from ml.datahandler import load_spacetrack_objects, build_features
from ml.dataset import split_by_object
from ml.inference import (load_checkpoint, load_pretrained_backbone, embed_objects,
                          cluster_embeddings, reconstruct_window)

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "rapport" / "figures"
RUN_DIR = ROOT / "outputs" / "ml" / "pretrain"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## tous les taux d'entraînement rencontrés doivent figurer dans la grille, sinon un
## modèle n'est jamais évalué à son propre point de fonctionnement.
MASK_GRID = [0.25, 0.30, 0.40, 0.45, 0.50, 0.65, 0.80]
KM_BUCKETS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.5), (0.5, 1.5), (1.5, 5.0), (5.0, np.inf)]
LEVEL_BUCKETS = [(0.0, 1.0), (1.0, 5.0), (5.0, 25.0), (25.0, 100.0), (100.0, np.inf)]
STEP_BUCKETS = [(0.05, 0.2), (0.2, 0.5), (0.5, 1.5), (1.5, np.inf)]
MAX_WIN_PER_OBJ = 8      # fenêtres évaluées par objet, réparties sur la série
EMBED_MAX_OBJ = 3000     # objets pour le clustering et la sonde
SEED = 0


# --------------------------------------------------------------------------- runs

def discover_runs():
    """Tous les runs de pretrain avec un best.pt, décrits par leur config."""
    runs = []
    for d in sorted(RUN_DIR.glob("*/")):
        ckpt = d / "checkpoints" / "best.pt"
        if not ckpt.exists():
            continue
        cfg = OmegaConf.load(d / ".hydra" / "config.yaml")
        head = torch.load(ckpt, map_location="cpu", weights_only=False)
        metrics = d / "metrics.jsonl"
        rows = [json.loads(l) for l in open(metrics)] if metrics.exists() else []
        runs.append({
            "name": d.name,
            "ckpt": ckpt,
            "cfg": cfg,
            "n_features": int(head["n_features"]),
            "dataset": str(cfg.data.get("dataset", "leo")),
            "scaler_kind": str(head.get("scaler_kind", "global")),
            "revin_win": bool(cfg.data.get("revin_per_window_norm", False)),
            "window": int(cfg.data.window_size),
            "patch": int(cfg.model.patch_size),
            "masking": float(cfg.model.masking_ratio),
            "blocks": int(cfg.model.encoder_n_blocks),
            "model": str(cfg.model.name),
            "epochs": len(rows),
            "train_curve": [r["train loss"] for r in rows],
            "val_curve": [r["val loss"] for r in rows],
            "val_logged": rows[-1]["val loss"] if rows else float("nan"),
            "val_best": min((r["val loss"] for r in rows), default=float("nan")),
        })
    return runs


def label(run, runs=None):
    """Étiquette courte et lisible d'un run dans les figures et tableaux.

    Deux runs peuvent partager exactement la même configuration (la référence à 500 époques et
    le point de grille à 200, par exemple) : on ajoute alors la date et le nombre d'époques,
    sans quoi la légende porterait deux fois la même entrée pour deux courbes différentes.
    """
    bits = [run["model"], f"W{run['window']}", f"P{run['patch']}", f"m{run['masking']:.2f}"]
    if run["blocks"] != 12:
        bits.append(f"b{run['blocks']}")
    if run["revin_win"]:
        bits.append("revin-fen")
    if run["scaler_kind"] == "per_obj":
        bits.append("norm-obj")
    base = " ".join(bits)
    if runs is not None and sum(1 for o in runs if label(o) == base) > 1:
        base += f" [{run['name'][5:10]}, {run['epochs']} ep]"
    return base


def _zero(w, feature_cols, cols):
    for c in cols:
        if c in feature_cols:
            w[feature_cols.index(c)] = 0.0
    return w


def candidate_weightings(cfg, feature_cols):
    """Les quatre pondérations successives de la loss, dans l'ordre chronologique.

    Les poids par canal n'ont été exposés dans la config que le 25/08 : avant, ils étaient
    codés en dur dans train.py, et pour partie jamais commités. Ils ne sont donc pas
    récupérables depuis les artefacts d'un run antérieur. On teste les quatre régimes qui se
    sont succédé et on retient celui qui reproduit la loss loggée : la pondération effective
    d'un run devient une grandeur mesurée plutôt que supposée.
    """
    n = len(feature_cols)
    out = {}
    out["1. dt seul a 0"] = _zero(np.ones(n), feature_cols, ["dt"])
    out["2. + cosM, sinM, sma_level a 0"] = _zero(
        np.ones(n), feature_cols, ["dt", "cosM", "sinM", "sma_level"])
    w3 = _zero(np.ones(n), feature_cols, ["dt", "cosM", "sinM", "sma_level"])
    for c in ("p_diff", "q_diff"):
        if c in feature_cols:
            w3[feature_cols.index(c)] = 0.25
    out["3. + p_diff, q_diff a 0.25"] = w3
    out["4. + increments a 0"] = _zero(
        np.ones(n), feature_cols,
        ["dt", "cosM", "sinM", "sma_level", "sma_diff", "p_diff", "q_diff"])
    if "dt_weight" in cfg.task:
        w5 = np.ones(n)
        mapping = [("dt", "dt_weight"), ("cosM", "M_weight"), ("sinM", "M_weight"),
                   ("sma_level", "sma_level_weight"), ("sma_diff", "sma_diff_weight"),
                   ("p_diff", "p_q_diff_weight"), ("q_diff", "p_q_diff_weight")]
        for col, key in mapping:
            if col in feature_cols:
                w5[feature_cols.index(col)] = float(cfg.task[key])
        out["config du run"] = w5
    return out


def loss_weights(cfg, feature_cols):
    """Pondération retenue : celle de la config quand elle existe, sinon la forme attestée
    la plus courante dans l'historique."""
    cand = candidate_weightings(cfg, feature_cols)
    return cand.get("config du run", cand["3. + p_diff, q_diff a 0.25"])


# --------------------------------------------------------------------------- données

def load_group(dataset):
    """Séries brutes, features et split, chargés une seule fois par dataset."""
    print(f"[data] chargement de {dataset}...", flush=True)
    objects = load_spacetrack_objects(ROOT / "data" / "raw" / "spacetrack", dataset)
    raw, feature_cols = {}, None
    for oid, df in objects.items():
        feats, feature_cols = build_features(df, spacetrack=True)
        raw[oid] = feats[feature_cols].to_numpy(np.float32)
    train_ids, val_ids = split_by_object(raw.keys(), 0.2, 42)
    print(f"[data] {len(raw)} objets, {len(val_ids)} en validation, {len(feature_cols)} canaux",
          flush=True)
    return raw, feature_cols, train_ids, val_ids


def normalize(raw, train_ids, kind, feature_cols):
    """Applique la normalisation du run. Retourne (per_obj, {norad: scale}) où scale est le
    vecteur qui ramène l'espace normalisé aux unités physiques."""
    per_obj, scales = {}, {}
    if kind == "per_obj":
        for oid, X in raw.items():
            sc = StandardScaler().fit(X)
            per_obj[oid] = sc.transform(X).astype(np.float32)
            scales[oid] = sc.scale_.astype(np.float32)
    else:
        sc = StandardScaler().fit(np.concatenate([raw[o] for o in train_ids]))
        for oid, X in raw.items():
            per_obj[oid] = sc.transform(X).astype(np.float32)
            scales[oid] = sc.scale_.astype(np.float32)
    return per_obj, scales


def object_windows(X, window, n_max=MAX_WIN_PER_OBJ):
    """Départs de fenêtres répartis sur toute la série."""
    if len(X) < window:
        return np.empty(0, dtype=int)
    n = min(n_max, max(1, (len(X) - window) // 8 + 1))
    return np.linspace(0, len(X) - window, n).astype(int)


def noise_floor(raw, ids, i_col):
    """Bruit haute fréquence par objet (MAD de la différence seconde), en unités du canal."""
    out = []
    for oid in ids:
        v = raw[oid][:, i_col].astype(np.float64)
        if len(v) < 50:
            continue
        d2 = np.diff(v, n=2)
        out.append(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))
    return np.array(out)


# --------------------------------------------------------------------------- coeur

@torch.no_grad()
def evaluate(mae, per_obj, scales, val_ids, feature_cols, window, patch, ratio, seed=SEED,
             levels=None):
    """Une passe d'évaluation à taux de masquage imposé.

    Retourne le détail par canal (espace normalisé) et, pour sma_local, l'erreur en km par
    patch masqué avec son amplitude propre — la seule métrique invariante à la longueur de
    fenêtre, donc la seule comparable entre runs.
    """
    old, mae.masking_ratio = mae.masking_ratio, ratio
    F = len(feature_cols)
    i_loc = feature_cols.index("sma_local")
    se = np.zeros(F)
    var = np.zeros(F)
    n_win = 0
    patches = []                       # (amplitude km, rmse km, |biais| km, residu km)

    torch.manual_seed(seed)
    for oid in val_ids:
        X = per_obj[oid]
        starts = object_windows(X, window)
        if not len(starts):
            continue
        xb = torch.from_numpy(np.stack([X[s:s + window].T for s in starts])).float().to(DEVICE)
        pred, target = mae(xb)
        B, Nm, _ = pred.shape
        pr = pred.reshape(B, Nm, F, patch).cpu().numpy()
        tg = target.reshape(B, Nm, F, patch).cpu().numpy()

        se += ((pr - tg) ** 2).mean(axis=(1, 3)).sum(axis=0)
        mu = tg.mean(axis=(1, 3), keepdims=True)
        var += ((tg - mu) ** 2).mean(axis=(1, 3)).sum(axis=0)
        n_win += B

        ## km = unité normalisée x sigma de la fenêtre (RevIN) x sigma du scaler amont
        s_win = mae.last_sigma[:, i_loc].cpu().numpy() if hasattr(mae, "last_sigma") else np.ones(B)
        km = (s_win * float(scales[oid][i_loc]))[:, None, None]
        t_km, p_km = tg[:, :, i_loc, :] * km, pr[:, :, i_loc, :] * km
        err = p_km - t_km
        bias = err.mean(axis=2)
        patches.append(np.stack([
            t_km.max(axis=2) - t_km.min(axis=2),                       # amplitude du patch
            np.sqrt((err ** 2).mean(axis=2)),                          # rmse total
            np.abs(bias),                                              # erreur de NIVEAU
            np.sqrt(((err - bias[..., None]) ** 2).mean(axis=2)),      # erreur de FORME
            t_km.std(axis=2),                                          # écart type cible
            np.full(t_km.shape, float(levels[oid]) if levels else np.nan)[:, :, 0],
        ], axis=-1).reshape(-1, 6))

    mae.masking_ratio = old
    return {"mse": se / n_win, "var": var / n_win, "n_windows": n_win,
            "patches": np.concatenate(patches) if patches else np.empty((0, 6))}


def km_table(patches):
    """Erreur de FORME par bucket d'amplitude du patch masqué.

    L'erreur totale mélange deux choses de nature différente : un décalage de niveau constant
    sur le patch, et une erreur sur la forme du signal. Seule la seconde est gouvernée par
    l'amplitude locale — d'où le skill calculé ici sur la forme seule. Le niveau est mesuré
    séparément par level_table, contre la grandeur qui le gouverne réellement.
    """
    amp, rmse, bias, resid, std, level = patches.T
    rows = []
    for lo, hi in KM_BUCKETS:
        m = (amp >= lo) & (amp < hi)
        if m.sum() < 20:
            continue
        rows.append({
            "bucket": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}",
            "n": int(m.sum()), "part": float(m.mean()),
            "rmse": float(np.sqrt((rmse[m] ** 2).mean())),
            "bias": float(np.sqrt((bias[m] ** 2).mean())),
            "resid": float(np.sqrt((resid[m] ** 2).mean())),
            "std_cible": float(np.sqrt((std[m] ** 2).mean())),
            "skill": float(1 - (resid[m] ** 2).mean() / max((std[m] ** 2).mean(), 1e-15)),
            "skill_total": float(1 - (rmse[m] ** 2).mean() / max((std[m] ** 2).mean(), 1e-15)),
        })
    return rows


def level_table(patches):
    """Erreur de NIVEAU par amplitude propre de l'objet (RMS de son sma_local, en km).

    Un patch plat appartenant à un objet en pleine décroissance porte une erreur de niveau
    énorme sans que son amplitude locale n'en dise rien : bucketer le biais par l'amplitude
    produit des non-monotonies dénuées de sens physique. La variable de découpage est mesurée
    sur les données BRUTES, donc identique pour tous les runs — sans quoi un run à
    normalisation par fenêtre serait découpé selon une grandeur différente des autres.
    """
    amp, rmse, bias, resid, std, level = patches.T
    rows = []
    for lo, hi in LEVEL_BUCKETS:
        m = (level >= lo) & (level < hi)
        if m.sum() < 20:
            continue
        rows.append({
            "bucket": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}",
            "n": int(m.sum()), "part": float(m.mean()),
            "biais": float(np.sqrt((bias[m] ** 2).mean())),
            "biais_median": float(np.median(bias[m])),
            "forme": float(np.sqrt((resid[m] ** 2).mean())),
        })
    return rows


def detect_steps(v, noise, k=4, floor=0.15):
    """Ruptures de niveau sur v (km) : médiane après moins médiane avant."""
    if len(v) < 3 * k:
        return []
    a = np.array([np.median(v[t:t + k]) - np.median(v[t - k:t]) for t in range(k, len(v) - k)])
    idx = np.where(np.abs(a) > max(floor, 5 * noise))[0] + k
    keep = []
    for t in idx[np.argsort(-np.abs(a[idx - k]))]:
        if all(abs(int(t) - u) > k for u in keep):
            keep.append(int(t))
    return [(t, float(a[t - k])) for t in keep]


@torch.no_grad()
def step_restitution(mae, per_obj, raw, scales, ids, feature_cols, window, patch, rng,
                     max_per_obj=3):
    """Le décodeur restitue-t-il les sauts, ou interpole-t-il ? Le patch qui contient la
    rupture est masqué de force ; la pente de d_pred sur d_true vaut 1 si le saut est
    restitué, 0 si le modèle lisse."""
    N = mae.num_patches
    n_mask = max(int(round(N * mae.masking_ratio)) - 1, 0)
    i_loc = feature_cols.index("sma_local")
    rows = []
    for oid in ids:
        X = per_obj[oid]
        if len(X) < window:
            continue
        s_km = float(scales[oid][i_loc])
        v = X[:, i_loc] * s_km
        nz = noise_floor(raw, [oid], i_loc)
        events = detect_steps(v, float(nz[0]) if len(nz) else 0.02)
        for t, _ in events[:max_per_obj]:
            start = int(np.clip(t - window // 2, 0, len(X) - window))
            p_event = (t - start) // patch
            if not 1 <= p_event <= N - 2:
                continue
            others = rng.choice([p for p in range(N) if p != p_event], size=n_mask, replace=False)
            masked = np.sort(np.concatenate([[p_event], others]))
            x = X[start:start + window].T
            recon, _ = reconstruct_window(mae, x, masked_patches=masked, device=DEVICE)
            a, b, h = p_event * patch, (p_event + 1) * patch, patch // 2
            d_true = float((np.median(x[i_loc, a + h:b]) - np.median(x[i_loc, a:a + h])) * s_km)
            d_pred = float((np.median(recon[i_loc, a + h:b]) - np.median(recon[i_loc, a:a + h])) * s_km)
            rows.append((d_true, d_pred))
    return np.array(rows) if rows else np.empty((0, 2))


def step_table(rows):
    if not len(rows):
        return {}, []
    d_true, d_pred = rows.T
    keep = np.abs(d_true) > 0.05
    d_true, d_pred = d_true[keep], d_pred[keep]
    if len(d_true) < 10:
        return {}, []
    glob = {
        "n": int(len(d_true)),
        "pente": float(np.sum(d_true * d_pred) / np.sum(d_true ** 2)),
        "correlation": float(np.corrcoef(d_true, d_pred)[0, 1]),
        "signe": float(np.mean(np.sign(d_pred) == np.sign(d_true))),
        "skill_vs_lisse": float(1 - np.mean((d_pred - d_true) ** 2) / np.mean(d_true ** 2)),
        "amplitude_p50": float(np.median(np.abs(d_true))),
    }
    per = []
    for lo, hi in STEP_BUCKETS:
        m = (np.abs(d_true) >= lo) & (np.abs(d_true) < hi)
        if m.sum() < 10:
            continue
        per.append({
            "bucket": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}",
            "n": int(m.sum()),
            "pente": float(np.sum(d_true[m] * d_pred[m]) / np.sum(d_true[m] ** 2)),
            "correlation": float(np.corrcoef(d_true[m], d_pred[m])[0, 1]),
            "signe": float(np.mean(np.sign(d_pred[m]) == np.sign(d_true[m]))),
        })
    return glob, per


# --------------------------------------------------------------------------- embeddings

## Deux familles de cibles, à ne pas confondre à la lecture.
##  - CONTROLE : grandeurs statiques directement présentes dans l'entrée (sma_level est un
##    canal, l'excentricité et l'inclinaison se déduisent de k,h,p,q par moyenne). Un R2 élevé
##    y signifie seulement que le CLS retient la moyenne de ce qu'il voit : c'est un test de
##    non-régression, pas une performance.
##  - DYNAMIQUE : grandeurs qui décrivent le COMPORTEMENT de l'objet, jamais données telles
##    quelles en entrée. C'est la famille qui compte, puisque l'aval — clusterisation par
##    comportement et détection de manoeuvres — porte sur elle.
PROBE_CONTROLE = ("sma moyen", "excentricite", "inclinaison")
PROBE_DYNAMIQUE = ("derive sma", "amplitude sma", "taux de rupture", "bruit propre")


def probe_targets(raw, feature_cols, ids):
    """Grandeurs physiques par objet, cibles de la sonde linéaire."""
    i_lvl, i_k, i_h = (feature_cols.index(c) for c in ("sma_level", "k", "h"))
    i_p, i_q, i_loc = (feature_cols.index(c) for c in ("p", "q", "sma_local"))
    out = defaultdict(list)
    for oid in ids:
        X = raw[oid].astype(np.float64)
        v = X[:, i_loc]
        out["sma moyen"].append(X[:, i_lvl].mean())
        out["excentricite"].append(np.sqrt(X[:, i_k] ** 2 + X[:, i_h] ** 2).mean())
        out["inclinaison"].append(2 * np.arctan(np.sqrt(X[:, i_p] ** 2 + X[:, i_q] ** 2)).mean())
        ## pente du demi-grand axe : proxy de la trainee
        out["derive sma"].append(np.polyfit(np.arange(len(X)), v, 1)[0] * 1000)
        out["amplitude sma"].append(v.std())
        ## proxy direct du "cet objet manoeuvre" : densité de ruptures de niveau
        nz = noise_floor(raw, [oid], i_loc)
        noise = float(nz[0]) if len(nz) else 0.02
        out["taux de rupture"].append(1000 * len(detect_steps(v, noise)) / max(len(v), 1))
        out["bruit propre"].append(noise)
    return {k: np.array(v) for k, v in out.items()}


## Cibles à queue lourde : l'amplitude du sma, la dérive et le bruit s'étalent sur plusieurs
## ordres de grandeur. Une poignée d'objets en décroissance rapide domine alors l'erreur
## quadratique et le R2 bascule d'un extrême à l'autre selon qu'ils tombent en validation ou
## non — mesuré : de -0,07 à -16,7 en passant de 600 à 2000 objets. On les compresse en log
## (signé pour la dérive, qui change de signe) et on reporte en plus un Spearman, insensible
## à la queue de distribution.
PROBE_LOG = ("amplitude sma", "bruit propre", "taux de rupture")
PROBE_LOG_SIGNE = ("derive sma",)


def _transform(name, y):
    if name in PROBE_LOG:
        return np.log1p(np.maximum(y, 0))
    if name in PROBE_LOG_SIGNE:
        return np.sign(y) * np.log1p(np.abs(y))
    return y


def linear_probe(emb, targets, ids, seed=42):
    """Régression ridge sur embedding gelé : R2 et Spearman en validation."""
    keep = [i for i, o in enumerate(ids) if o in emb]
    if len(keep) < 100:
        return {}
    X = StandardScaler().fit_transform(np.stack([emb[ids[i]] for i in keep]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keep))
    cut = int(.8 * len(keep))
    tr, va = perm[:cut], perm[cut:]
    scores = {}
    for name, y_all in targets.items():
        y = _transform(name, y_all[keep].astype(np.float64))
        if np.std(y) < 1e-12:
            continue
        pred = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[va])
        scores[name] = float(r2_score(y[va], pred))
        scores[name + " (rho)"] = float(spearmanr(y[va], pred).statistic)
    return scores


def cluster_quality(emb):
    """HDBSCAN à hyperparamètres fixés : structure trouvée dans la représentation."""
    X, labels, norads = cluster_embeddings(emb, min_cluster_size=15, min_samples=15)
    keep = labels != -1
    n_clusters = int(len(set(labels[keep])))
    out = {"n_objets": int(len(labels)), "n_clusters": n_clusters,
           "fraction_bruit": float((~keep).mean())}
    if n_clusters >= 2 and keep.sum() > 10:
        out["silhouette"] = float(silhouette_score(X[keep], labels[keep]))
    return out, X, labels


# --------------------------------------------------------------------------- figures

def fig_curves(runs, out):
    """Courbes de validation. Les niveaux ne sont comparables qu'à l'intérieur d'un groupe
    de même jeu de features et de même normalisation — d'où le découpage en panneaux."""
    groups = [("11 canaux (avant le 18/08)", lambda r: r["n_features"] == 11),
              ("12 canaux, LEO", lambda r: r["n_features"] == 12 and r["dataset"] == "leo"),
              ("12 canaux, Starlink", lambda r: r["n_features"] == 12 and r["dataset"] == "starlink")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, pred) in zip(axes, groups):
        sel = [r for r in runs if pred(r) and r["epochs"] > 3]
        for r in sorted(sel, key=lambda r: r["name"]):
            ax.plot(r["val_curve"], lw=1.3, label=label(r, runs))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("époque")
        ax.set_yscale("log")
        ax.grid(alpha=.25)
        if sel:
            ax.legend(fontsize=6.5, loc="upper right")
    axes[0].set_ylabel("val loss (échelle log)")
    fig.suptitle("Courbes de validation — les niveaux ne sont pas comparables entre panneaux",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_channels(results, feature_cols, out):
    """Skill par canal, un groupe de barres par run."""
    names = [r["label"] for r in results]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(feature_cols))
    w = 0.8 / max(len(results), 1)
    for i, r in enumerate(results):
        skill = np.array(r["channels"]["skill"])
        ax.bar(x + i * w, np.clip(skill, -0.5, 1), w, label=names[i])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(feature_cols, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("skill  $1 - \\mathrm{MSE}/\\mathrm{Var}$")
    ax.set_title("Reconstruction par canal (valeurs sous $-0{,}5$ tronquées)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_km(results, noise_km, out):
    """Erreur en km sur sma_local par amplitude du patch masqué, avec le plancher de bruit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ## axe catégoriel : le dernier bucket est ouvert (">5 km"), lui affecter une abscisse
    ## numérique reviendrait à inventer une amplitude moyenne.
    names = [b["bucket"] for b in max((r["km"] for r in results if r.get("km")),
                                      key=len, default=[])]
    for r in results:
        rows = {b["bucket"]: b for b in r.get("km", [])}
        if not rows:
            continue
        xs = [i for i, n in enumerate(names) if n in rows]
        axes[0].plot(xs, [rows[names[i]]["resid"] for i in xs], "o-", lw=1.4, ms=4,
                     label=r["label"])
        axes[1].plot(xs, [rows[names[i]]["skill"] for i in xs], "o-", lw=1.4, ms=4,
                     label=r["label"])
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20)
        ax.set_xlabel("amplitude du patch masqué (km)")
        ax.grid(alpha=.25, which="both")
    axes[0].axhspan(0, float(np.percentile(noise_km, 90)), color="grey", alpha=.25,
                    label="bruit TLE (p90)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("erreur de forme (km)")
    axes[0].set_title("Erreur de forme, hors décalage de niveau")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(-600, 1.5)
    axes[1].set_ylabel("skill")
    axes[1].set_title("Skill sur la forme : $0$ = aussi bon que prédire une constante")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_mask_matrix(results, out):
    """Chaque modèle évalué à tous les taux : sépare la difficulté de la tâche de la qualité
    du modèle."""
    sel = [r for r in results if r.get("mask_transfer")]
    if not sel:
        return
    M = np.array([[r["mask_transfer"][f"{m:.2f}"]["rmse_km"] for m in MASK_GRID] for r in sel])
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(sel) + 3))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(MASK_GRID)))
    ax.set_xticklabels([f"{m:.2f}" for m in MASK_GRID])
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([r["label"] for r in sel], fontsize=8)
    ax.set_xlabel("taux de masquage à l'évaluation")
    for i, r in enumerate(sel):
        for j, m in enumerate(MASK_GRID):
            own = abs(m - r["masking"]) < 0.03
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if own else "normal",
                    color="white" if own else "black")
    ax.set_title("Erreur de forme sur sma_local (km) — gras : taux d'entraînement")
    fig.colorbar(im, ax=ax, label="RMSE (km)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_steps(results, out):
    """Pente de restitution des sauts, par bucket d'amplitude."""
    sel = [r for r in results if r.get("steps_per_bucket")]
    if not sel:
        return
    buckets = [b["bucket"] for b in sel[0]["steps_per_bucket"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(buckets))
    w = 0.8 / len(sel)
    for i, r in enumerate(sel):
        by = {b["bucket"]: b for b in r["steps_per_bucket"]}
        axes[0].bar(x + i * w, [by.get(b, {}).get("pente", np.nan) for b in buckets], w,
                    label=r["label"])
        axes[1].bar(x + i * w, [by.get(b, {}).get("signe", np.nan) for b in buckets], w,
                    label=r["label"])
    for ax, ylab, ref in ((axes[0], "pente $d_{pred}/d_{true}$", 1.0),
                          (axes[1], "fraction de signes corrects", 0.5)):
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("amplitude du saut (km)")
        ax.set_ylabel(ylab)
        ax.axhline(ref, color="k", ls="--", lw=.9)
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(0, color="tab:red", ls=":", lw=1.1)
    axes[0].set_title("1 = saut restitué, 0 = interpolation lisse")
    axes[1].set_title("0,5 = hasard")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_embed(results, out):
    """Qualité de la représentation : structure de clustering et sonde linéaire."""
    sel = [r for r in results if r.get("clusters")]
    if not sel:
        return
    names = [r["label"] for r in sel]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    y = np.arange(len(sel))

    axes[0].barh(y, [r["clusters"]["n_clusters"] for r in sel], color="tab:blue")
    axes[0].set_title("clusters HDBSCAN (MCS=15, MS=15)")
    axes[1].barh(y, [r["clusters"]["fraction_bruit"] for r in sel], color="tab:orange")
    axes[1].set_title("fraction d'objets non clusterisés")
    for ax in axes[:2]:
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.grid(axis="x", alpha=.25)

    order = list(PROBE_CONTROLE) + list(PROBE_DYNAMIQUE)
    targets = [t for t in order if any(t in r.get("probe", {}) for r in sel)]  # R2 seuls
    w = 0.8 / max(len(targets), 1)
    for j, t in enumerate(targets):
        axes[2].barh(y + j * w, [r.get("probe", {}).get(t, np.nan) for r in sel], w, label=t)
    axes[2].set_yticks(y + 0.4 - w / 2)
    axes[2].set_yticklabels(names, fontsize=8)
    axes[2].set_xlabel("$R^2$ en validation")
    axes[2].set_title("sonde linéaire : contrôle (statique) vs dynamique")
    axes[2].set_xlim(-0.1, 1)
    axes[2].axvline(0, color="k", lw=.8)
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_examples(pairs, per_obj_by_kind, scales_by_kind, feature_cols, norads, out):
    """Mêmes objets, mêmes fenêtres, deux normalisations : le biais de niveau se voit."""
    i_loc = feature_cols.index("sma_local")
    fig, axes = plt.subplots(len(norads), len(pairs), figsize=(7 * len(pairs), 2.7 * len(norads)),
                             squeeze=False)
    for row, norad in enumerate(norads):
        for col, run in enumerate(pairs):
            mae, cfg = run["mae"], run["cfg"]
            W, P = run["window"], run["patch"]
            per_obj = per_obj_by_kind[run["scaler_kind"]]
            scales = scales_by_kind[run["scaler_kind"]]
            X = per_obj[norad]
            start = max(0, len(X) // 2 - W // 2)
            x = X[start:start + W].T
            recon, masked = reconstruct_window(mae, x, device=DEVICE, seed=1)
            s_km = float(scales[norad][i_loc])
            t = np.arange(start, start + W)
            ax = axes[row][col]
            for p in masked:
                ax.axvspan(start + p * P, start + (p + 1) * P, color="grey", alpha=.16, lw=0)
            pred = np.full(W, np.nan)
            for p in masked:
                pred[p * P:(p + 1) * P] = recon[i_loc, p * P:(p + 1) * P] * s_km
            ax.plot(t, x[i_loc] * s_km, color="black", lw=1.2)
            ax.plot(t, pred, color="tab:red", lw=1.5)
            if row == 0:
                ax.set_title(run["label"], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"norad {norad}\nsma_local (km)", fontsize=8)
    fig.suptitle("Reconstruction sur les patchs masqués — noir : réel, rouge : reconstruit",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- tables LaTeX

def _f(x, n=3, na="---"):
    """Nombre au format français, virgule décimale, pour insertion directe dans le .tex."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return f"{x:.{n}f}".replace(".", ",")


def _esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write_tables(out, feature_cols, dataset="leo"):
    """Écrit les tableaux du rapport directement depuis les métriques mesurées.

    Aucun chiffre du rapport n'est recopié à la main : le .tex fait \\input de ces fragments.
    C'est la seule façon de garantir que le texte et le JSON ne divergent pas.
    """
    runs = out.get(dataset, [])
    if not runs:
        return

    def dump(name, body):
        path = FIG_DIR / f"pretrain_tab_{name}.tex"
        path.write_text(body, encoding="utf-8")
        print(f"[tex] {path.name}")

    ## --- inventaire des runs
    rows = []
    for r in out["runs"]:
        if r["n_features"] != 12 or r["dataset"] != dataset or r["epochs"] < 3:
            continue
        rows.append(f"{_esc(r['name'])} & {r['window']} & {r['patch']} & "
                    f"{_f(r['masking'], 2)} & {r['blocks']} & {r['epochs']} & "
                    f"{_f(r['val_best'], 4)} \\\\")
    dump("runs", "\\begin{tabular}{lccccrr}\n\\toprule\nRun & Fenêtre & Patch & Masquage & "
         "Blocs & Époques & Val loss \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n")

    ## --- pondérations reconstituées
    rows = []
    for r in runs:
        c = r.get("ponderations_candidates", {})
        if not c:
            continue
        rows.append(f"{_esc(r['label'])} & {_f(r.get('val_best'), 4)} & "
                    + " & ".join(_f(c.get(k), 4) for k in sorted(c)) +
                    f" & {_esc(r.get('ponderation_retenue', ''))[:12]} \\\\")
    if rows:
        keys = sorted({k for r in runs for k in r.get("ponderations_candidates", {})})
        dump("ponderations",
             "\\begin{tabular}{l" + "c" * (len(keys) + 2) + "}\n\\toprule\nRun & Loggée & "
             + " & ".join(_esc(k) for k in keys) + " & Retenu \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- skill par canal (canaux à variance non nulle uniquement)
    keep = [j for j, c in enumerate(feature_cols) if c not in ("sma_level",)]
    rows = []
    for r in runs:
        ch = r.get("channels")
        if not ch:
            continue
        rows.append(f"{_esc(r['label'])} & "
                    + " & ".join(_f(ch["skill"][j], 2) for j in keep) + " \\\\")
    if rows:
        dump("canaux", "\\begin{tabular}{l" + "c" * len(keep) + "}\n\\toprule\nRun & "
             + " & ".join("\\texttt{" + _esc(feature_cols[j]) + "}" for j in keep)
             + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- part de la loss par canal
    rows = []
    for r in runs:
        ch = r.get("channels")
        if not ch:
            continue
        rows.append(f"{_esc(r['label'])} & "
                    + " & ".join(_f(100 * ch["part_loss"][j], 1) for j in keep) + " \\\\")
    if rows:
        dump("part_loss", "\\begin{tabular}{l" + "c" * len(keep) + "}\n\\toprule\nRun & "
             + " & ".join("\\texttt{" + _esc(feature_cols[j]) + "}" for j in keep)
             + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- erreur de forme par amplitude
    names = [b["bucket"] for b in max((r.get("km", []) for r in runs), key=len, default=[])]
    rows = []
    for r in runs:
        by = {b["bucket"]: b for b in r.get("km", [])}
        if not by:
            continue
        rows.append(f"{_esc(r['label'])} & "
                    + " & ".join(_f(by[n]["resid"], 3) if n in by else "---" for n in names)
                    + " \\\\")
    if rows:
        dump("forme", "\\begin{tabular}{l" + "c" * len(names) + "}\n\\toprule\nRun & "
             + " & ".join(_esc(n) for n in names) + " \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- matrice de masquage
    grid = sorted({k for r in runs for k in r.get("mask_transfer", {})})
    rows = []
    for r in runs:
        mt = r.get("mask_transfer")
        if not mt:
            continue
        cells = []
        for g in grid:
            v = _f(mt[g]["rmse_km"], 3)
            cells.append("\\textbf{" + v + "}" if abs(float(g) - r["masking"]) < 0.03 else v)
        rows.append(f"{_esc(r['label'])} & " + " & ".join(cells) + " \\\\")
    if rows:
        dump("masquage", "\\begin{tabular}{l" + "c" * len(grid) + "}\n\\toprule\nRun & "
             + " & ".join(_f(float(g), 2) for g in grid) + " \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- sonde linéaire et clustering
    order = [t for t in list(PROBE_CONTROLE) + list(PROBE_DYNAMIQUE)
             if any(t in r.get("probe", {}) for r in runs)]
    rows = []
    for r in runs:
        pr, cl = r.get("probe"), r.get("clusters")
        if not pr:
            continue
        rows.append(f"{_esc(r['label'])} & "
                    + " & ".join(_f(pr.get(t), 2) for t in order)
                    + f" & {cl['n_clusters'] if cl else '---'} & "
                    + (_f(cl["fraction_bruit"], 2) if cl else "---") + " \\\\")
    if rows:
        dump("sonde", "\\begin{tabular}{l" + "c" * (len(order) + 2) + "}\n\\toprule\nRun & "
             + " & ".join(_esc(t) for t in order) + " & Clusters & Bruit \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    ## --- restitution des ruptures
    bucks = [b["bucket"] for b in max((r.get("steps_per_bucket", []) for r in runs),
                                      key=len, default=[])]
    rows = []
    for r in runs:
        by = {b["bucket"]: b for b in r.get("steps_per_bucket", [])}
        g = r.get("steps") or {}
        if not by:
            continue
        rows.append(f"{_esc(r['label'])} & {_f(g.get('pente'), 2)} & "
                    + " & ".join(_f(by[n]["pente"], 2) if n in by else "---" for n in bucks)
                    + " \\\\")
    if rows:
        dump("ruptures", "\\begin{tabular}{lc" + "c" * len(bucks) + "}\n\\toprule\nRun & "
             "Global & " + " & ".join(_esc(n) for n in bucks) + " \\\\\n\\midrule\n"
             + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="curves,channels,km,mask,steps,embed,examples")
    ap.add_argument("--datasets", default="leo,starlink")
    ap.add_argument("--min-epochs", type=int, default=100,
                    help="runs plus courts exclus des comparaisons (entraînement non abouti)")
    ap.add_argument("--embed-obj", type=int, default=EMBED_MAX_OBJ,
                    help="objets utilisés pour le clustering et la sonde linéaire")
    ap.add_argument("--max-obj", type=int, default=400,
                    help="objets de validation utilisés pour les métriques de reconstruction")
    args = ap.parse_args()
    modules = set(args.only.split(","))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    print(f"[runs] {len(runs)} checkpoints trouvés")
    out = {"runs": [{k: v for k, v in r.items()
                     if k not in ("ckpt", "cfg", "train_curve", "val_curve")} for r in runs]}

    if "curves" in modules:
        fig_curves(runs, FIG_DIR / "pretrain_training_curves.png")
        print("[fig] pretrain_training_curves.png")

    rng = np.random.default_rng(SEED)
    featcols = {}          # les colonnes dépendent du dataset : ne pas les partager
    for dataset in args.datasets.split(","):
        sel = [r for r in runs if r["dataset"] == dataset and r["n_features"] == 12
               and r["epochs"] >= args.min_epochs]
        skipped = [r for r in runs if r["dataset"] == dataset and r["n_features"] == 12
                   and r["epochs"] < args.min_epochs]
        if skipped:
            print(f"[{dataset}] écartés, moins de {args.min_epochs} époques : "
                  + ", ".join(f"{r['name']} ({r['epochs']})" for r in skipped))
        if not sel:
            continue
        raw, feature_cols, train_ids, val_ids = load_group(dataset)
        featcols[dataset] = feature_cols
        i_loc = feature_cols.index("sma_local")
        noise_km = noise_floor(raw, val_ids[:1500], i_loc)
        out[f"{dataset}_bruit_km"] = {f"p{p}": float(np.percentile(noise_km, p))
                                      for p in (10, 50, 90)}
        print(f"[{dataset}] bruit TLE sma : p50 {np.median(noise_km) * 1000:.0f} m, "
              f"p90 {np.percentile(noise_km, 90) * 1000:.0f} m")

        norm_cache = {}
        for kind in {r["scaler_kind"] for r in sel}:
            norm_cache[kind] = normalize(raw, train_ids, kind, feature_cols)

        ## amplitude propre de chaque objet, en km : mesurée sur les données brutes, donc
        ## identique pour tous les runs quelle que soit leur normalisation. C'est elle qui
        ## gouverne l'erreur de niveau, pas l'agitation locale du patch.
        levels = {o: float(np.sqrt((raw[o][:, i_loc].astype(np.float64) ** 2).mean()))
                  for o in val_ids}
        eval_ids = [o for o in val_ids if len(raw[o]) >= 256][:args.max_obj]
        targets = probe_targets(raw, feature_cols, val_ids) if "embed" in modules else None

        results, loaded = [], []
        for r in sel:
            print(f"[{dataset}] {label(r, runs)} ({r['epochs']} ep)", flush=True)
            mae, cfg, _, _ = load_checkpoint(r["ckpt"], DEVICE)
            per_obj, scales = norm_cache[r["scaler_kind"]]
            res = {"name": r["name"], "label": label(r, runs), "epochs": r["epochs"],
                   "masking": r["masking"],
                   "window": r["window"], "patch": r["patch"], "blocks": r["blocks"],
                   "revin_win": r["revin_win"], "scaler_kind": r["scaler_kind"],
                   "val_logged": r["val_logged"], "val_best": r["val_best"]}
            ids = [o for o in eval_ids if len(raw[o]) >= r["window"]]

            if {"channels", "km"} & modules:
                ev = evaluate(mae, per_obj, scales, ids, feature_cols,
                              r["window"], r["patch"], r["masking"], levels=levels)
                w = loss_weights(r["cfg"], feature_cols)
                ## sma_level est constant sur une fenêtre : sa variance de cible est nulle et
                ## le skill n'y est pas défini. On le marque plutôt que d'afficher -1e14.
                skill = np.where(ev["var"] > 1e-6, 1 - ev["mse"] / np.maximum(ev["var"], 1e-15),
                                 np.nan)
                share = ev["mse"] * w / max((ev["mse"] * w).sum(), 1e-15)
                res["loss_recalculee"] = float((ev["mse"] * w).sum() / w.sum())
                ## quelle pondération reproduit la loss loggée ? cf. candidate_weightings
                cands = {k: float((ev["mse"] * v).sum() / v.sum())
                         for k, v in candidate_weightings(r["cfg"], feature_cols).items()}
                res["ponderations_candidates"] = cands
                ## on compare a la MEILLEURE val loss, pas a la derniere : le checkpoint
                ## evalue est best.pt, et pour un run non converge les deux different.
                ref = r["val_best"]
                res["ponderation_retenue"] = min(
                    cands, key=lambda k: abs(cands[k] / max(ref, 1e-9) - 1))
                res["ecart_ponderation"] = float(cands[res["ponderation_retenue"]] / ref - 1)
                res["channels"] = {"mse": ev["mse"].tolist(), "var": ev["var"].tolist(),
                                   "skill": skill.tolist(), "poids": w.tolist(),
                                   "part_loss": share.tolist()}
                res["km"] = km_table(ev["patches"])
                res["niveau"] = level_table(ev["patches"])
                res["n_windows"] = ev["n_windows"]

            if "mask" in modules:
                res["mask_transfer"] = {}
                for m in MASK_GRID:
                    e = evaluate(mae, per_obj, scales, ids[:150], feature_cols,
                                 r["window"], r["patch"], m)
                    w = loss_weights(r["cfg"], feature_cols)
                    res["mask_transfer"][f"{m:.2f}"] = {
                        "loss": float((e["mse"] * w).sum() / w.sum()),
                        "rmse_km": float(np.sqrt((e["patches"][:, 3] ** 2).mean())),
                    }

            if "steps" in modules:
                rows = step_restitution(mae, per_obj, raw, scales, ids[:200], feature_cols,
                                        r["window"], r["patch"], rng)
                g, per = step_table(rows)
                res["steps"], res["steps_per_bucket"] = g, per

            if "embed" in modules:
                backbone, _, _, _ = load_pretrained_backbone(r["ckpt"], DEVICE)
                sub = {o: per_obj[o] for o in list(per_obj)[:args.embed_obj]
                       if len(per_obj[o]) >= r["window"]}
                emb = embed_objects(backbone, sub, DEVICE, window_size=r["window"],
                                    stride=max(r["window"] // 2, 1))
                cl, _, _ = cluster_quality(emb)
                res["clusters"] = cl
                res["probe"] = linear_probe(emb, probe_targets(raw, feature_cols, list(sub)),
                                            list(sub))
                print(f"    clusters {cl['n_clusters']}, bruit {cl['fraction_bruit']:.2f}, "
                      f"sonde {res['probe']}", flush=True)

            results.append(res)
            loaded.append({**r, "mae": mae, "label": label(r, runs)})

        out[dataset] = results
        pre = f"pretrain_{dataset}"
        if "channels" in modules:
            fig_channels(results, feature_cols, FIG_DIR / f"{pre}_channel_skill.png")
        if "km" in modules:
            fig_km(results, noise_km, FIG_DIR / f"{pre}_km_error.png")
        if "mask" in modules:
            fig_mask_matrix(results, FIG_DIR / f"{pre}_mask_matrix.png")
        if "steps" in modules:
            fig_steps(results, FIG_DIR / f"{pre}_steps.png")
        if "embed" in modules:
            fig_embed(results, FIG_DIR / f"{pre}_embeddings.png")
        if "examples" in modules and len(loaded) >= 2:
            pairs = loaded[-2:]
            norads = [o for o in eval_ids
                      if all(len(raw[o]) >= p["window"] for p in pairs)][:3]
            fig_examples(pairs, {k: v[0] for k, v in norm_cache.items()},
                         {k: v[1] for k, v in norm_cache.items()},
                         feature_cols, norads, FIG_DIR / f"{pre}_examples.png")
        print(f"[fig] figures {pre}_* écrites")

    for ds, cols in featcols.items():
        if ds in out:
            write_tables(out, cols, ds)

    with open(FIG_DIR / "pretrain_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[ok] {FIG_DIR / 'pretrain_metrics.json'}")


if __name__ == "__main__":
    main()
