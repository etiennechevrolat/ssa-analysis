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

def build_doris_samples(norad_id, labels):
    """
    Créer une liste de manoeuvres pour un sat sous la forme : 
    (time_index, maneuver_type, maneuver_intensity) avec des bosses triangulaires autour de l'epoch.
    Le time_index est fixé au time index du tle spacetrack le plus proche de l'epoch originale de manoeuvre, dans l'index créé par load_doris_object.
    Il y a 3 types d'intensité de manoeuvres : on calcule la moyenne, variance des delta_v du satellite puis :
    les manoeuvres faibles dans mean +- sigma, 
    les manoeuvres moyennes dans mean +- 2sigma, 
    les manoeuvres fortes dans mean +- 3sigma
    """
    object_labels = labels[labels['norad_id'] == norad_id] 
    
    samples = []
    return samples