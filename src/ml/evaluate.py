import numpy as np
from collections import defaultdict

## Paramètres d'évaluation

detection_treshold = 0.1 ## manoeuvre détectée pour proba_pred > treshold
matching_tolerance = 6 ## 6jours de tolérance pour détection de manoeuvres

## Noeuds de manoeuvres réelle
pol_nodes= ('ID', 'AD', 'IK')



### EVALUTATION DU LOCALIZER: 
# Nécessite une mise des prédictions d'un tableau de taille N par batch avec des logits -> à des événements ponctuels dans les deux directions


## Mise des labels au format pour tous les objets :  {oid : {'EW' : [], 'NS' : []}}
def gt_events_from_labels(labels, # DataFrame : ObjectID, Direction, Node, TimeIndex, Type
                          objects_ids,
                          nodes=pol_nodes
    ):
    out={}
    for oid in objects_ids:
        sub = labels[labels['ObjectID'] == oid]
        event = {}
        for direction in ('EW', 'NS'):
            t = sub[(sub['Direction'] == direction) & (sub['Node'].isin(nodes))]['TimeIndex']
            event[direction] = sorted(int(x) for x in t.to_numpy())
        out[oid] = event 
    return out 


## Mise des prédictions au format events pour UN objet {'EW' : [], 'NS' : []}
def _run_centers(mask : np.ndarray)-> list[int]:
    """Récupère un mask de type [F,T,T,T,F,T,T,F] et renvoie les indices des centres des suites de True
    """
    idx = np.flatnonzero(mask) # indices ou mask est true
    if idx.size == 0:
        return []
    # on coupe idx en différents runs, ie. la ou deux indices ne se suivent pas.
    runs = np.split(idx, np.flatnonzero(np.diff(idx)> 1) + 1)
    # on renvoie le milieu de ces différents runs
    return [run[len(run)//2]  for run in runs]

def extract_events(scores : np.ndarray, # tableau (L, 2) avec une colonne par direction EW/NS
                treshold: float | dict[str, float] = detection_treshold,
                is_logits: bool=False
    )-> dict[str, list[int]] :
    """(L,2) scores -> {"EW":[t..], "NS":[t..]} après seuillage + collapse au centre.
    treshold accepte un flottant unique ou un seuil par direction {"EW": .., "NS": ..} :
    les manoeuvres in-plane et out-of-plane n'ont pas le même rapport signal/bruit, et
    un seuil commun dégrade la direction la moins bien séparée.
    """
    if is_logits:
        scores = 1/ (1 + np.exp(-scores))
    events = {}
    for col,name in (0,'EW'), (1,'NS'):
        th = treshold[name] if isinstance(treshold, dict) else treshold
        mask = scores[:, col] > th
        events[name] = _run_centers(mask)
    return events
 
## Matching entre les labels et prédictions 
def match_events(gt_idx : list[int],  # liste d'instants de vraies manoeuvres
                pred_idx: list[int],  # liste d'instants de manoeuvres prédites
                tolerance=matching_tolerance)-> dict:
    """Matche deux listes triées d'indices selon UNE direction pour UN objet)."""

    matched = set() # indices de pred_idx déjà appariés
    tp, fn = 0,0
    distances= []
    for g in gt_idx: 
        best = None ## on retient une prédiction pour chaque vraie manoeuvre de gt_idx
        for j,p in enumerate(pred_idx):
            if j in matched:
                continue
            if abs(p-g) <= tolerance : 
                if best is None or abs(p-g) < abs(pred_idx[best] - g):
                    best = j
        if best is None : 
            fn +=1
        else : 
            tp += 1
            matched.add(best)
            distances.append(pred_idx[best] - g)
    fp = len(pred_idx) - len(matched)
    return {'tp' : tp, 'fp' : fp, 'fn' : fn, 'distances' :distances}

## Calcul des métriques de performance du modèle 
def metrics(tp, fp, fn, distances):
    precision = tp /(tp + fp)  if (tp + fp) > 0 else 0.0
    recall = tp/(tp + fn) if (tp + fn) > 0 else 0.0

    def fbeta(beta, tp, fp,fn):
        beta2 = beta*beta
        denom = (1+beta2)*tp + beta2 * fn + fp 
        return (1+beta2) * tp / denom if denom > 0 else 0.0
    f1 = fbeta(1, tp,fp,fn)
    f2 = fbeta(2, tp,fp,fn)
    d = np.asarray(distances, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(d**2))) if d.size >0 else 0.0

    return {"precision": precision, "recall": recall,
            "f1": f1, "f2": f2, "distances" :d , "rmse": rmse,
            "tp": tp, "fp": fp, "fn": fn}

## On passe 
def evaluate_predictions(gt_events: dict, 
                         pred_events : dict,
                         tolerance=matching_tolerance
                         ):
    """Accumule la confusion matrix pour tous les objets sur les deux direction, 
    puis applique les métriques sur la confusion matrix globale"""

    tp = fp= fn = 0
    distances=[]
    for oid in gt_events: 
        for d in ('EW','NS'):
            gt_idx = gt_events[oid][d]
            pred_idx = pred_events[oid][d]
            r =match_events(gt_idx,pred_idx,tolerance) 
            tp += r['tp']
            fp += r['fp']
            fn += r['fn']
            distances += r['distances']
    return metrics(tp, fp, fn, distances)


                        


