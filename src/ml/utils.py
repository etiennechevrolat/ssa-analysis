import numpy as np 
import torch
from torch import nn
import pandas as pd 
import os 

def json_to_csv(data_path):
    json_path = os.path.join(data_path, 'labels.json')
    df = pd.read_json(json_path)
    df_filtered = df[df['orbit_class']=='LEO'] 
    
    csv_fil_path = os.path.join(data_path, 'leo_maneuvers.csv')
    df_filtered.to_csv(csv_fil_path, index=False)

def parquet_to_csv(data_path):
    parquet_path = os.path.join(data_path, 'orbital_parameters.parquet')
    df = pd.read_parquet(parquet_path)
    
    csv_fil_path = os.path.join(data_path, 'leo_doris_orbital_params_example.csv')
    df.to_csv(csv_fil_path, index=False)


def to_device(batch, device):
    """
    Gère les deux types de modèles, 
    soit localizer et les labels y sont des simples tenseurs , 
    soit classifier et y sont des dict avec des clés 'node' et 'class'. 
    """
    if isinstance(batch, dict):
        return {k: v.to(device) for k,v in batch.items()}
    return batch.to(device)

def compute_class_weights(dataset, field, n_classes, device):
    """
    field = position de la classe rare dans le tuple d'index : pour le node_type : 3"""

    counts = np.zeros(n_classes, dtype=np.float64)
    for sample in dataset.index: # oid, t, dir, node, class
        counts [sample[field]] += 1
    counts=np.maximum(counts, 1)

    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)

class MaskedChannelMSE(nn.Module):
    def __init__(self, channel_weights):
        super().__init__()  
        self.register_buffer("w", channel_weights)

    def forward(self, pred, target):
        B,N,_ = pred.shape 
        F = self.w.numel()
        d2= (pred-target).reshape(B,N,F,-1).pow(2).mean(dim=(0,1,3))
        return (d2 * self.w).sum() / self.w.sum()


def main() : 
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, '..', '..', 'data', 'parsed', 'labelled_leo_DORIS', 'train')
    parquet_to_csv(data_path)

if __name__=="__main__":
    main()