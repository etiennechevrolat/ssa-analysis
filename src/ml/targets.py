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

INTENSITY_QUANTILES = (0.5, 0.95)


def build_doris_targets(df, norad_id, labels, half_width=6, quantiles=INTENSITY_QUANTILES):
    """
    Cible (L, 2, 3) : bosses triangulaires autour des manoeuvres, avec
    axe 1 = type de manoeuvre  (0 = in-track, 1 = cross-track)
    axe 2 = intensité du delta_v (0 = faible, 1 = moyenne, 2 = forte)

    L = longueur de la série de l'objet (df), pas le nombre de manoeuvres.
    Le time_index d'une manoeuvre est celui du TLE spacetrack le plus proche de son epoch,
    dans l'index créé par load_doris_objects (les labels hors fenêtre TLE y ont TimeIndex = NaN).

    Les 3 classes d'intensité sont découpées sur les quantiles des delta_v du satellite :
    faibles jusqu'au quantile 0.5 (ie dures à différencier du bruit pour le modèle),
    moyennes entre les quantiles 0.5 et 0.95,
    fortes au delà du quantile 0.95.
    """
    L = len(df)
    Y = np.zeros((L, 2, 3), dtype=np.float32)
    w = half_width

    ## les labels sans TimeIndex (hors fenêtre TLE de l'objet) ne sont pas projetables sur la série
    object_labels = labels[(labels['norad_id'] == norad_id) & labels['TimeIndex'].notna()].copy()
    if object_labels.empty:
        return Y
    object_labels['TimeIndex'] = object_labels['TimeIndex'].astype(int)

    delta_v = object_labels['delta_v'].to_numpy(float)
    if np.isnan(delta_v).all(): ## ex. norads 22076 et 22823 : aucun delta_v renseigné
        return Y
    q_lo, q_hi = np.nanquantile(delta_v, quantiles)

    for row in object_labels.itertuples(index=False):
        type_idx = doris_type_to_index.get(row.maneuver_type)
        if type_idx is None or np.isnan(row.delta_v): ## maneuver_type absent/radial, ou delta_v manquant
            continue

        if row.delta_v < q_lo:      # noise maneuver
            intensity_idx = 0
        elif row.delta_v < q_hi:    # classic maneuver
            intensity_idx = 1
        else:                       # strong maneuver
            intensity_idx = 2

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