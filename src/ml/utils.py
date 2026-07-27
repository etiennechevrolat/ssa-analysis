

def to_device(batch, device):
    """
    Gère les deux types de modèles, 
    soit localizer et les labels y sont des simples tenseurs , 
    soit classifier et y sont des dict avec des clés 'node' et 'class'. 
    """
    if isinstance(batch, dict):
        return {k: v.to(device) for k,v in batch.items()}
    return batch.to(device)
