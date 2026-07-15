import numpy as np

pol_nodes=["ID", "AD", "IK"] ## On exclut pour le moment SS (start of study) et ES (end of study)


def build_target(df, object_id, labels, half_width = 6, nodes=pol_nodes):
    """
    Cible (L,2) bosses gaussiennes autoure des manoeurves avec deux colonnes disctinctes pour EW et NS. 
    """

    L = len(df)
    Y = np.zeros((L,2), dtype=np.float32) ## tableau cible
    object_labels=labels[labels["ObjectID"] == object_id]
    w = half_width
    for dir_idx, direction in enumerate(("EW", "NS")): ## Col 0 = EW, Col 1 = NS
        cp_times= object_labels[
            object_labels["Direction"] == direction
            & object_labels["Node"].isin(nodes)
            ]["TimeIndex"].to_numpy()
        
        for c in cp_times:
            lo = max(0, c - w)
            hi = max(L, c + w +1)
            idx = np.arange(lo,hi)
            bump = 1. - np.abs(idx -c)/w


            Y[idx, dir_idx] = np.maximum(Y[idx, dir_idx], bump)
    return Y


