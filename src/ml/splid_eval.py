"""Évaluation officielle SPLID.

Port du NodeDetectionEvaluator du splid-devkit (ARCLab-MIT), qui a servi à
scorer le concours : appariement des noeuds prédits aux noeuds vrais avec une
tolérance de +/- 6 pas de temps (12 h), par objet et par direction (EW/NS),
puis Precision / Recall / F2 / RMSE calculés sur la matrice de confusion globale.

Deux modes :
  - localizer_only=True  : un match temporel suffit (évalue la détection seule) ;
  - localizer_only=False : mode concours, il faut aussi Node et Type identiques
    (évalue la chaîne localizer + classifier).

Fournit aussi la construction d'une soumission au format du concours
(ObjectID, TimeIndex, Direction, Node, Type) à partir des checkpoints
localizer/classifier du repo, comme dans le repo gagnant (splid-challenge).
"""

import numpy as np
import pandas as pd
import torch

from ml.targets import pol_nodes, type_to_index, class_to_index
from ml.evaluate import extract_events, detection_treshold

TOLERANCE = 6  # +/- 6 TimeIndex = 12h, valeur officielle du concours

index_to_type = {v: k for k, v in type_to_index.items()}
index_to_class = {v: k for k, v in class_to_index.items()}


## ---------------------------------------------------------------- évaluateur

def evaluate_object(gt_object, p_object, tolerance=TOLERANCE, localizer_only=False):
    """Apparie les noeuds d'UN objet (mêmes règles que le devkit) :
    pour chaque noeud vrai (dans l'ordre du fichier), on prend le premier noeud
    prédit non encore apparié à moins de `tolerance` pas dans la même direction.
    TP si (Node, Type) coïncident (toujours vrai en localizer_only), FP sinon.
    Les prédictions restantes sont des FP, les noeuds vrais non appariés des FN.
    """
    gt_object = gt_object[gt_object['Direction'] != 'ES']
    p_object = p_object[p_object['Direction'] != 'ES'].sort_values('TimeIndex')

    p_times = p_object['TimeIndex'].to_numpy()
    p_dirs = p_object['Direction'].to_numpy()
    p_nodes = p_object['Node'].to_numpy()
    p_types = p_object['Type'].to_numpy()
    matched = np.zeros(len(p_object), dtype=bool)

    tp = fp = fn = 0
    distances = []
    for _, gt_row in gt_object.iterrows():
        candidates = np.flatnonzero(
            (np.abs(p_times - gt_row['TimeIndex']) <= tolerance)
            & (p_dirs == gt_row['Direction'])
            & ~matched
        )
        if candidates.size == 0:
            fn += 1
            continue
        j = candidates[0]
        matched[j] = True
        if localizer_only or (p_nodes[j] == gt_row['Node'] and p_types[j] == gt_row['Type']):
            tp += 1
            distances.append(int(p_times[j] - gt_row['TimeIndex']))
        else:
            fp += 1
    fp += int((~matched).sum())
    return tp, fp, fn, distances


def score(ground_truth, participant, tolerance=TOLERANCE, localizer_only=False):
    """Score global sur tous les objets de ground_truth.
    ground_truth / participant : DataFrames (ObjectID, TimeIndex, Direction, Node, Type).
    Renvoie un dict {precision, recall, f2, rmse, tp, fp, fn}.
    """
    tp = fp = fn = 0
    distances = []
    for oid in ground_truth['ObjectID'].unique():
        gt_object = ground_truth[ground_truth['ObjectID'] == oid]
        p_object = participant[participant['ObjectID'] == oid]
        r = evaluate_object(gt_object, p_object, tolerance, localizer_only)
        tp, fp, fn = tp + r[0], fp + r[1], fn + r[2]
        distances += r[3]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) else 0.0
    rmse = float(np.sqrt(np.mean(np.array(distances, float) ** 2))) if distances else 0.0
    return {"precision": precision, "recall": recall, "f2": f2, "rmse": rmse,
            "tp": tp, "fp": fp, "fn": fn}


## ------------------------------------------------- construction de soumission

@torch.no_grad()
def predict_scores(model, X, history, future, device, batch_size=256):
    """X (L,F) normalisé -> probas (L,2) via fenêtre glissante paddée."""
    Xpad = np.pad(X, ((history, future), (0, 0)), mode='constant')
    W = history + future + 1
    windows = np.lib.stride_tricks.sliding_window_view(Xpad, W, axis=0)  # (L,F,W)
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch_size])).float()
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def classify_events(model, X, times, history, future, device):
    """Classifie les fenêtres centrées sur les instants `times` d'un objet.
    Renvoie (nodes, types) : listes de labels ('ID'/'AD'/'IK', 'NK'/'CK'/'EK'/'HK').
    """
    Xpad = np.pad(X, ((history, future), (0, 0)), mode='constant')
    W = history + future + 1
    x = np.stack([Xpad[t:t + W].T for t in times])  # (N,F,W)
    logits = model(torch.from_numpy(x).float().to(device))
    nodes = [index_to_type[i] for i in logits['node'].argmax(1).cpu().numpy()]
    types = [index_to_class[i] for i in logits['class'].argmax(1).cpu().numpy()]
    return nodes, types


def build_submission(per_obj, object_ids, localizer, history, future, device,
                     classifier=None, threshold=detection_treshold,
                     clf_history=None, clf_future=None):
    """Chaîne complète localizer (+ classifier) -> DataFrame de soumission.

    per_obj : {oid : [X normalisé, .]} (cf. dataset.build_arrays + scaler)
    Chaque objet reçoit en outre ses deux noeuds SS à TimeIndex 0 (convention du
    dataset : début de période d'étude), typés par le classifier si disponible.

    Les deux modèles peuvent avoir été entraînés sur des fenêtres différentes :
    clf_history/clf_future décrivent celle du classifier (par défaut celle du localizer).
    """
    clf_history = history if clf_history is None else clf_history
    clf_future = future if clf_future is None else clf_future

    rows = []
    for oid in object_ids:
        X = per_obj[oid][0]
        scores = predict_scores(localizer, X, history, future, device)
        events = extract_events(scores, treshold=threshold)
        for direction in ('EW', 'NS'):
            times = [0] + list(events[direction])  # SS + détections
            if classifier is not None:
                nodes, types = classify_events(classifier, X, times, clf_history, clf_future, device)
            else:
                nodes, types = ['ID'] * len(times), ['NK'] * len(times)
            nodes[0] = 'SS'
            for t, node, type_ in zip(times, nodes, types):
                rows.append((oid, int(t), direction, node, type_))
    return pd.DataFrame(rows, columns=['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type'])


def ground_truth_for(labels, object_ids, keep_ss=True):
    """Sous-ensemble des labels pour les objets donnés, au format de l'évaluateur.
    keep_ss=False restreint aux noeuds de manoeuvre (ID/AD/IK), pour évaluer la
    détection seule sans compter les noeuds conventionnels SS.
    """
    gt = labels[labels['ObjectID'].isin(object_ids)].copy()
    if not keep_ss:
        gt = gt[gt['Node'].isin(pol_nodes)]
    return gt
