import numpy as np
from collections import defaultdict

from ml.targets import doris_types, intensity_labels, DELTA_V_THRESHOLD

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

def extract_events(scores : np.ndarray, # tableau (L, n_canaux), une colonne par canal
                treshold: float | dict[str, float] = detection_treshold,
                is_logits: bool=False,
                channels : tuple[str, ...] = ('EW', 'NS')
    )-> dict[str, list[int]] :
    """(L, n_canaux) scores -> {canal : [t..]} après seuillage + collapse au centre.
    treshold accepte un flottant unique ou un seuil par canal {"EW": .., "NS": ..} :
    les manoeuvres in-plane et out-of-plane n'ont pas le même rapport signal/bruit, et
    un seuil commun dégrade la direction la moins bien séparée.
    channels nomme les colonnes : EW/NS pour SPLID, les 4 canaux type/intensité pour DORIS.
    """
    if is_logits:
        scores = 1/ (1 + np.exp(-scores))
    events = {}
    for col, name in enumerate(channels):
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
                         tolerance=matching_tolerance,
                         channels : tuple[str, ...] = ('EW', 'NS')
                         ):
    """Accumule la confusion matrix pour tous les objets sur tous les canaux,
    puis applique les métriques sur la confusion matrix globale"""

    tp = fp= fn = 0
    distances=[]
    for oid in gt_events:
        ## tolerance peut être un dict {oid : tol} : la cadence TLE varie d'un facteur ~4
        ## entre satellites, donc une tolérance en indices ne vaut pas la même durée partout.
        tol = tolerance[oid] if isinstance(tolerance, dict) else tolerance
        for d in channels:
            gt_idx = gt_events[oid][d]
            pred_idx = pred_events[oid][d]
            r =match_events(gt_idx,pred_idx,tol)
            tp += r['tp']
            fp += r['fp']
            fn += r['fn']
            distances += r['distances']
    return metrics(tp, fp, fn, distances)


### EVALUATION DU FINETUNING DORIS
## Même protocole que le localizer, mais les canaux ne sont plus EW/NS : ce sont les 4
## combinaisons (type de manoeuvre, intensité) dans l'ordre d'aplatissement de la cible (2,2).

doris_channels = tuple(f"{maneuver_type}/{intensity}"
                       for maneuver_type in doris_types for intensity in intensity_labels)


def gt_events_doris(labels,        # DataFrame rendu par load_doris_objects : norad_id, TimeIndex, maneuver_type, delta_v
                    norad_ids,
                    threshold=DELTA_V_THRESHOLD,
                    detection_only=False
    ):
    """{norad : {canal : [TimeIndex..]}}
    Les filtres doivent être EXACTEMENT ceux de build_doris_targets (TimeIndex non NaN,
    maneuver_type connu, delta_v renseigné) : sinon la vérité terrain compte des manoeuvres
    que la cible n'a jamais marquées, et le rappel plafonne sans raison visible.
    detection_only=True fusionne tout dans un canal unique 'maneuver'.
    """
    out = {}
    for oid in norad_ids:
        sub = labels[(labels['norad_id'] == oid) & labels['TimeIndex'].notna()
                     & labels['delta_v'].notna() & labels['maneuver_type'].isin(doris_types)]
        if detection_only:
            out[oid] = {'maneuver' : sorted(int(x) for x in sub['TimeIndex'].to_numpy())}
            continue
        event = {}
        for maneuver_type in doris_types:
            for i, intensity in enumerate(intensity_labels):
                is_strong = (sub['delta_v'] >= threshold) if i == 1 else (sub['delta_v'] < threshold)
                t = sub[(sub['maneuver_type'] == maneuver_type) & is_strong]['TimeIndex']
                event[f"{maneuver_type}/{intensity}"] = sorted(int(x) for x in t.to_numpy())
        out[oid] = event
    return out


                        


