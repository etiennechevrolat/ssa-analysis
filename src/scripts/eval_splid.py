"""Évaluation SPLID complète + génération des figures du rapport.

Usage :
    PYTHONPATH=src .venv/bin/python src/scripts/eval_splid.py

Le script :
  1. retrouve les derniers runs de chaque modèle dans outputs/ml/{localizer,classifier} ;
  2. trace les courbes d'entraînement (val F2 par epoch) de tous les localizers ;
  3. pour le meilleur localizer : sweep du seuil de détection (P/R/F2),
     score officiel splid-devkit (détection seule et chaîne complète avec classifier),
     exemples de prédictions vs labels sur des objets de validation ;
  4. matrices de confusion du classifier (node et type de propulsion) ;
  5. figure du fossé de domaine SPLID vs TLE SpaceTrack (cadence d'échantillonnage).

Toutes les figures sont écrites dans rapport/figures/ (préfixe splid_) et les
métriques dans rapport/figures/splid_metrics.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from ml.datahandler import load_splid_objects
from ml.dataset import build_arrays, split_by_object
from ml.evaluate import extract_events, gt_events_from_labels, evaluate_predictions
from ml.inference import load_checkpoint
from ml.splid_eval import score, build_submission, ground_truth_for, predict_scores

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "rapport" / "figures"
DATA_DIR = ROOT / "data" / "raw" / "splid_dataset" / "training"
LABELS = ROOT / "data" / "raw" / "splid_dataset" / "train_label.csv"

MODELS = ["cnn_lstm1", "small_cnn", "small_lstm", "naive_baseline"]


def last_run_per_model(task_dir):
    """{model_name : run_dir} en prenant le run le plus récent de chaque modèle."""
    runs = {}
    for run in sorted(task_dir.glob("*/")):
        cfg_path = run / ".hydra" / "config.yaml"
        if not (run / "checkpoints" / "best.pt").exists():
            continue
        name = OmegaConf.load(cfg_path).model.name
        runs[name] = run  # tri chronologique -> le dernier écrase
    return runs


def read_metrics(run_dir):
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return pd.DataFrame(rows)


def val_scores_per_object(model, cfg, per_obj, val_ids, device):
    """Probas (L,2) du localizer pour chaque objet de validation."""
    h, f = cfg.data.history, cfg.data.future
    return {oid: predict_scores(model, per_obj[oid][0], h, f, device) for oid in val_ids}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    FIG_DIR.mkdir(exist_ok=True)
    results = {}

    loc_runs = last_run_per_model(ROOT / "outputs" / "ml" / "localizer")
    clf_runs = last_run_per_model(ROOT / "outputs" / "ml" / "classifier")
    print("localizer runs:", {k: v.name for k, v in loc_runs.items()})
    print("classifier runs:", {k: v.name for k, v in clf_runs.items()})

    ## ------------------------------------------------ 1. courbes d'entraînement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in MODELS:
        if name not in loc_runs:
            continue
        m = read_metrics(loc_runs[name])
        axes[0].plot(m["epoch"], m["val loss"], marker="o", label=name)
        axes[1].plot(m["epoch"], m["val f2"], marker="o", label=name)
        results.setdefault("val_f2_max", {})[name] = float(m["val f2"].max())
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val loss (BCE)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val F2 (tolérance 6 pas)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Localizer : courbes de validation par modèle")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "splid_training_curves.png", dpi=150)
    plt.close(fig)

    ## ------------------------------------------------ données + meilleur localizer
    best_name = max(results["val_f2_max"], key=results["val_f2_max"].get)
    results["best_model"] = best_name
    print("meilleur localizer :", best_name, results["val_f2_max"])

    loc_model, loc_cfg, mean, scale = load_checkpoint(
        loc_runs[best_name] / "checkpoints" / "best.pt", device)

    objects, labels = load_splid_objects(DATA_DIR, LABELS)
    per_obj, feature_cols = build_arrays(objects, labels, half_width=loc_cfg.task.half_width)
    for oid in per_obj:
        per_obj[oid][0] = ((per_obj[oid][0] - mean) / scale).astype(np.float32)
    train_ids, val_ids = split_by_object(per_obj.keys(), loc_cfg.data.val_split, loc_cfg.seed)
    print(f"{len(train_ids)} objets train / {len(val_ids)} objets val")

    scores_val = val_scores_per_object(loc_model, loc_cfg, per_obj, val_ids, device)

    ## ------------------------------------------------ 2. sweep du seuil de détection
    gt_events = gt_events_from_labels(labels, val_ids)
    thresholds = np.linspace(0.02, 0.9, 45)
    sweep = []
    for th in thresholds:
        pred_events = {oid: extract_events(s, treshold=th) for oid, s in scores_val.items()}
        m = evaluate_predictions(gt_events, pred_events)
        sweep.append((th, m["precision"], m["recall"], m["f2"]))
    sweep = np.array(sweep)
    best_i = int(np.argmax(sweep[:, 3]))
    best_th = float(sweep[best_i, 0])
    results["threshold_sweep"] = {"best_threshold": best_th,
                                  "precision": float(sweep[best_i, 1]),
                                  "recall": float(sweep[best_i, 2]),
                                  "f2": float(sweep[best_i, 3])}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sweep[:, 0], sweep[:, 1], label="precision")
    ax.plot(sweep[:, 0], sweep[:, 2], label="recall")
    ax.plot(sweep[:, 0], sweep[:, 3], label="F2", lw=2.5)
    ax.axvline(best_th, color="k", ls="--", alpha=0.5,
               label=f"seuil optimal = {best_th:.2f} (F2={sweep[best_i,3]:.3f})")
    ax.set_xlabel("seuil de détection sur la probabilité")
    ax.set_ylabel("score (validation)")
    ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f"{best_name} : précision / rappel / F2 en fonction du seuil")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "splid_threshold_sweep.png", dpi=150)
    plt.close(fig)

    ## ------------------------------------------------ 3. score officiel splid-devkit
    clf_model = clf_cfg = None
    if "cnn_lstm1" in clf_runs:
        clf_model, clf_cfg, _, _ = load_checkpoint(
            clf_runs["cnn_lstm1"] / "checkpoints" / "best.pt", device)

    h, f = loc_cfg.data.history, loc_cfg.data.future
    submission = build_submission(per_obj, val_ids, loc_model, h, f, device,
                                  classifier=clf_model, threshold=best_th)

    # détection seule : noeuds de manoeuvre uniquement (sans SS), match temporel
    gt_man = ground_truth_for(labels, val_ids, keep_ss=False)
    sub_man = submission[submission["Node"] != "SS"]
    results["official_localizer_only"] = score(gt_man, sub_man, localizer_only=True)

    # chaîne complète : tous les noeuds (avec SS), match (temps, Node, Type)
    gt_full = ground_truth_for(labels, val_ids, keep_ss=True)
    results["official_full"] = score(gt_full, submission, localizer_only=False)
    print("officiel détection seule :", results["official_localizer_only"])
    print("officiel chaîne complète :", results["official_full"])

    ## ------------------------------------------------ 4. exemples de prédictions
    # objets val avec le plus de manoeuvres
    n_man = {oid: sum(len(v) for v in gt_events[oid].values()) for oid in val_ids}
    examples = sorted(val_ids, key=lambda o: -n_man[o])[:3]
    for oid in examples:
        df = objects[oid]
        s = scores_val[oid]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
        axes[0].plot(df["TimeIndex"], df["Semimajor Axis (m)"] / 1000, lw=0.8)
        axes[0].set_ylabel("a (km)")
        axes[1].plot(df["TimeIndex"], df["Inclination (deg)"], lw=0.8, color="tab:orange")
        axes[1].set_ylabel("i (deg)")
        axes[2].plot(df["TimeIndex"], s[:, 0], lw=0.9, label="proba EW")
        axes[2].plot(df["TimeIndex"], s[:, 1], lw=0.9, label="proba NS")
        axes[2].axhline(best_th, color="k", ls="--", alpha=0.5, label="seuil")
        axes[2].set_ylabel("probabilité de noeud")
        axes[2].set_xlabel("TimeIndex (pas de 2 h)")
        for t in gt_events[oid]["EW"]:
            for ax in axes:
                ax.axvline(t, color="tab:blue", alpha=0.25)
        for t in gt_events[oid]["NS"]:
            for ax in axes:
                ax.axvline(t, color="tab:orange", alpha=0.25, ls=":")
        axes[2].legend(loc="upper right")
        fig.suptitle(f"Objet {oid} (validation) : labels (traits verticaux) vs sortie du localizer")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"splid_example_{oid}.png", dpi=150)
        plt.close(fig)
    results["example_objects"] = examples

    ## ------------------------------------------------ 5. confusion du classifier
    if clf_model is not None:
        from ml.dataset import build_classifier_arrays
        from ml.targets import type_to_index, class_to_index
        from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

        per_obj_c, _ = build_classifier_arrays(objects, labels)
        for oid in per_obj_c:
            per_obj_c[oid][0] = ((per_obj_c[oid][0] - mean) / scale).astype(np.float32)

        hc, fc = clf_cfg.data.history, clf_cfg.data.future
        Wc = hc + fc + 1
        y_true = {"node": [], "class": []}
        y_pred = {"node": [], "class": []}
        with torch.no_grad():
            for oid in val_ids:
                X, samples = per_obj_c[oid]
                if not samples:
                    continue
                Xpad = np.pad(X, ((hc, fc), (0, 0)), mode="constant")
                x = np.stack([Xpad[t:t + Wc].T for t, _, _, _ in samples])
                logits = clf_model(torch.from_numpy(x).float().to(device))
                y_true["node"] += [s[2] for s in samples]
                y_true["class"] += [s[3] for s in samples]
                y_pred["node"] += logits["node"].argmax(1).cpu().tolist()
                y_pred["class"] += logits["class"].argmax(1).cpu().tolist()

        node_names = list(type_to_index)
        class_names = list(class_to_index)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for ax, head, names in ((axes[0], "node", node_names), (axes[1], "class", class_names)):
            cm = confusion_matrix(y_true[head], y_pred[head], labels=range(len(names)))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(names)), names)
            ax.set_yticks(range(len(names)), names)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            acc = accuracy_score(y_true[head], y_pred[head])
            f1m = f1_score(y_true[head], y_pred[head], average="macro", zero_division=0)
            ax.set_title(f"{'type de noeud' if head=='node' else 'type de propulsion'}\n"
                         f"acc={acc:.3f}, F1 macro={f1m:.3f}")
            ax.set_xlabel("prédit"); ax.set_ylabel("vrai")
            results[f"classifier_{head}"] = {"acc": float(acc), "f1_macro": float(f1m)}
        fig.suptitle("Classifier cnn_lstm1 : matrices de confusion (validation)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "splid_classifier_confusion.png", dpi=150)
        plt.close(fig)

    ## ------------------------------------------------ 6. fossé de domaine SpaceTrack
    # SPLID fournit les éléments OSCULATEURS d'une simulation haute fidélité : le
    # demi-grand axe y porte des oscillations à courte période (12 h / 24 h en GEO)
    # de plusieurs km. Les TLE SpaceTrack sont des éléments MOYENS au sens SGP4 :
    # ces oscillations en ont été retirées. C'est le coeur du fossé de domaine.
    geo_parquets = sorted((ROOT / "data" / "raw").glob("RADUGA_ALL_*.parquet"))
    if geo_parquets:
        raw = pd.read_parquet(geo_parquets[-1])
        norad = raw["norad"].value_counts().idxmax()
        sat = raw[raw["norad"] == norad].sort_values("epoch").drop_duplicates("epoch")
        dts = pd.to_datetime(sat["epoch"]).diff().dt.total_seconds().dropna() / 3600.0

        # dispersion des incréments de a, sur tout le dataset de chaque domaine
        std_splid = np.array([np.diff(df["Semimajor Axis (m)"].to_numpy() / 1000).std()
                              for df in objects.values()])
        std_st = []
        for _, sub in raw.groupby("norad"):
            sub = sub.sort_values("epoch").drop_duplicates("epoch")
            if len(sub) >= 50:
                std_st.append(np.diff(sub["sma"].to_numpy(float)).std())
        std_st = np.array(std_st)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # (a) séries temporelles sur 5 jours, centrées : oscillation vs lissage
        n5 = 60  # 5 jours à 2 h
        a_splid = objects[val_ids[0]]["Semimajor Axis (m)"].to_numpy()[:n5] / 1000
        t_splid = np.arange(n5) * 2 / 24
        ep = pd.to_datetime(sat["epoch"])
        m5 = ep < ep.iloc[0] + pd.Timedelta(days=5)
        a_st = sat["sma"].to_numpy(float)[m5.to_numpy()]
        t_st = (ep[m5.to_numpy()] - ep.iloc[0]).dt.total_seconds().to_numpy() / 86400
        axes[0].plot(t_splid, a_splid - a_splid.mean(), marker=".", ms=3,
                     label="SPLID (osculateur, simulé)")
        axes[0].plot(t_st, a_st - a_st.mean(), marker="o", ms=4,
                     label="TLE SpaceTrack (moyen, réel)")
        axes[0].set_xlabel("temps (jours)")
        axes[0].set_ylabel("$a - \\overline{a}$ (km)")
        axes[0].set_title("Demi-grand axe centré sur 5 jours")
        axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

        # (b) cadence d'échantillonnage
        axes[1].hist(dts.clip(0, 48), bins=48, color="tab:red", alpha=0.75)
        axes[1].axvline(2.0, color="k", ls="--", label="cadence SPLID (2 h)")
        axes[1].set_xlabel("intervalle entre TLE consécutifs (h)")
        axes[1].set_ylabel("nombre de TLE")
        axes[1].set_title(f"Cadence SpaceTrack (NORAD {norad})\nmédiane = {dts.median():.1f} h")
        axes[1].legend(fontsize=8)

        # (c) dispersion des incréments, par objet
        axes[2].hist(std_splid, bins=30, alpha=0.7,
                     label=f"SPLID ({len(std_splid)} objets)\nmédiane {np.median(std_splid):.2f} km")
        axes[2].hist(std_st, bins=30, alpha=0.7,
                     label=f"SpaceTrack ({len(std_st)} objets)\nmédiane {np.median(std_st):.3f} km")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("$\\sigma(\\Delta a)$ par objet (km, échelle log)")
        axes[2].set_ylabel("nombre d'objets")
        axes[2].set_title("Amplitude des incréments de $a$")
        axes[2].legend(fontsize=8)

        fig.suptitle("Fossé de domaine : éléments osculateurs simulés (SPLID) vs éléments moyens TLE (SpaceTrack)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "splid_domain_gap.png", dpi=150)
        plt.close(fig)
        results["domain_gap"] = {
            "norad": int(norad),
            "median_dt_h": float(dts.median()),
            "median_std_da_splid_km": float(np.median(std_splid)),
            "median_std_da_spacetrack_km": float(np.median(std_st)),
            "ratio": float(np.median(std_splid) / np.median(std_st)),
        }

    with open(FIG_DIR / "splid_metrics.json", "w") as fjson:
        json.dump(results, fjson, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
