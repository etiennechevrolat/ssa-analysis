import numpy as np 
import torch


def to_device(batch, device):
    """
    Gère les deux types de modèles, 
    soit localizer et les labels y sont des simples tenseurs , 
    soit classifier et y sont des dict avec des clés 'node' et 'class'. 
    """
    if isinstance(batch, dict):
        return {k: v.to(device) for k,v in batch.items()}
    return batch.to(device)

def compute_class_weights(dataset, field, n_node_classes, device):
    """
    field = position de la classe rare dans le tuple d'index : pour le node_type : 3"""

    counts = np.zeros(n_node_classes, dtype=np.float64)
    for sample in dataset.index: # oid, t, dir, node, class
        counts [sample[field]] += 1
    counts=np.maximum(counts, 1)

    w = counts.sum() / (n_node_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)
