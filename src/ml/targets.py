import numpy as np

pol_nodes=["ID", "AD", "IK"] ## On exclut pour le moment SS (start of study) et ES (end of study)

type_to_index = {node_type : i for i,node_type in enumerate(pol_nodes)}

class_to_index = {
    'NK' : 0, ## not station-keeping
    'CK' : 1, ## station keeping with chemical propulsion
    'EK' : 2, ## station keeping with electric propulsion
    'HK' : 3, ## station keeping with hybrid propulsion
}


def build_target(df, object_id, labels, half_width = 6, nodes=pol_nodes):
    """
    Cible (L,2) bosses triangulaires autour des manoeurves avec deux colonnes disctinctes pour EW et NS. 
    """
    L = len(df)
    Y = np.zeros((L,2), dtype=np.float32) ## tableau cible
    object_labels=labels[labels["ObjectID"] == object_id]
    w = half_width
    for dir_idx, direction in enumerate(("EW", "NS")): ## Col 0 = EW, Col 1 = NS
        cp_times= object_labels[
            (object_labels["Direction"] == direction)
            & (object_labels["Node"].isin(nodes))
            ]["TimeIndex"].to_numpy()
        
        for c in cp_times:
            lo = max(0, c - w)
            hi = min(L, c + w +1)
            idx = np.arange(lo,hi)
            bump = 1. - np.abs(idx -c)/w

            Y[idx, dir_idx] = np.maximum(Y[idx, dir_idx], bump)
    return Y



def build_classifier_samples(object_id, labels, nodes=pol_nodes): 
    """
    Créer une liste des manoeuvres pour un sat sous forme de liste de (time_index, direction(prédite par localizer), node, class_type)
    """
    object_labels = labels[labels['ObjectID'] == object_id]

    samples = []

    for class_type in class_to_index:
        for dir_idx, direction in enumerate(("EW", "NS")): 
            for node_type in type_to_index:
                ## on récupère comme ça les TimeIndex des noeuds de type : type_node, class_node dans la direction  : direction.
                cp_times = object_labels[
                    (object_labels['Direction'] == direction) & 
                    (object_labels['Type'] == class_type) &
                    (object_labels["Node"] == node_type)
                    ]['TimeIndex']
                for c in cp_times:
                    samples.append((c, dir_idx,  type_to_index[node_type], class_to_index[class_type]))
    return samples

doris_types = ('in-track', 'cross-track') ## 'radial' (2 occurrences) et les types manquants sont ignorés
doris_type_to_index = {maneuver_type : i for i, maneuver_type in enumerate(doris_types)}

## Seuils absolus en m/s sur le delta_v, partagés par les deux types de manoeuvre.
## Choix de seuils fixes plutôt que de quantiles : les quantiles refittés par split de train
## variaient d'un facteur 190 sur le q50 cross-track (distribution bimodale, population basse
## concentrée sur quelques satellites), et ~40% des labels tombaient à +-25% du seuil.
## Ici 0.1 m/s marque la fin de la population in-track de routine (son décile 9 est à 0.1117)
## et se situe dans le creux de la distribution : ~2% des labels seulement sont proches du seuil.
## Un tuple plus long rajoute des classes d'intensité sans autre changement.
DELTA_V_THRESHOLDS = (0.1,)
intensity_labels = ('faible', 'forte')


def delta_v_to_intensity(delta_v, thresholds=DELTA_V_THRESHOLDS):
    """
    Classe d'intensité d'une manoeuvre à partir de son delta_v (m/s).
    Avec DELTA_V_THRESHOLDS = (0.1,) : 0 = faible (< 0.1 m/s), 1 = forte (>= 0.1 m/s).
    La borne appartient à la classe supérieure.
    """
    return np.searchsorted(thresholds, delta_v, side='right')


def build_doris_targets(df, norad_id, labels, half_width=6, thresholds=DELTA_V_THRESHOLDS):
    """
    Cible (L, 2, C) : bosses triangulaires autour des manoeuvres, avec
    axe 1 = type de manoeuvre    (0 = in-track, 1 = cross-track)
    axe 2 = intensité du delta_v (C = len(thresholds) + 1 classes, cf. intensity_labels)

    L = longueur de la série de l'objet (df), pas le nombre de manoeuvres.
    Le time_index d'une manoeuvre est celui du TLE spacetrack le plus proche de son epoch,
    dans l'index créé par load_doris_objects (les labels hors fenêtre TLE y ont TimeIndex = NaN).

    Les classes d'intensité sont découpées sur des seuils absolus en delta_v (DELTA_V_THRESHOLDS),
    identiques pour les deux types de manoeuvre : 'forte' désigne donc le même effet physique
    en in-track et en cross-track, et les seuils ne dépendent pas du split train/val.
    """
    L = len(df)
    Y = np.zeros((L, len(doris_types), len(thresholds) + 1), dtype=np.float32)
    w = half_width

    ## les labels sans TimeIndex (hors fenêtre TLE de l'objet) ne sont pas projetables sur la série
    object_labels = labels[(labels['norad_id'] == norad_id) & labels['TimeIndex'].notna()].copy()
    if object_labels.empty:
        return Y
    object_labels['TimeIndex'] = object_labels['TimeIndex'].astype(int)

    for row in object_labels.itertuples(index=False):
        type_idx = doris_type_to_index.get(row.maneuver_type)
        if type_idx is None or np.isnan(row.delta_v): ## maneuver_type absent/radial, ou delta_v manquant
            continue

        intensity_idx = delta_v_to_intensity(row.delta_v, thresholds)

        c = row.TimeIndex
        lo = max(0, c - w)
        hi = min(L, c + w + 1)
        idx = np.arange(lo, hi)
        bump = 1. - np.abs(idx - c) / w

        Y[idx, type_idx, intensity_idx] = np.maximum(Y[idx, type_idx, intensity_idx], bump)

    return Y


from pathlib import Path
import os 
from ml.datahandler import load_doris_objects
def main():
    base  = Path.cwd()
    data_dir =  os.path.join(base, 'data', 'parsed', 'labelled_leo_DORIS')
    data_path = os.path.join(data_dir, 'train', 'leo_doris_orbital_params.csv')
    labels_path = os.path.join(data_dir,  'leo_maneuvers_label.csv')
    objects, labels = load_doris_objects(data_path, labels_path)
    Y = build_doris_targets(objects[20436], norad_id=20436, labels=labels)
    print(Y.shape, Y.sum(axis=0))

if __name__ == "__main__" : 
    main()