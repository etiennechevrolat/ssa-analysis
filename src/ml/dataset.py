from ml.targets import build_target
from ml.datahandler import build_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


def build_arrays(objects, labels, half_width=6):
    """
    chaque objet -> (X: (L,F), Y: (L,2))
    retourne per_obj 
    """
    per_obj, feature_cols = {}, None
    for oid, df in objects.items():
        df_feat, fearure_cols = build_features(df)
        X = df_feat[feature_cols].to_numpy(np.float32)
        Y = build_target(df, oid, labels, half_width=half_width)
        per_obj[oid] = [X,Y]
    return per_obj, feature_cols

def split_by_object(object_ids, val_split=0.2, seed=42):
    """
    Split par objet du dataset et train/val
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(object_ids))
    rng.shuffle(ids)
    n_val=int(len(ids) * val_split)

    return(list(ids[n_val:]), list(ids[:n_val]))

def fit_scaler_on_train(per_obj, train_ids): 
    """Ajuste le scaler sur le train seulement.
    """
    ## fitter le scaler = trouver nu et sigma correspondant aux valeurs des features dans train_ids
    scaler=StandardScaler().fit(np.concatenate([per_obj[o][0] for o in train_ids])) 
    for oid in per_obj:
        ## applique x = (x-nu)/sigma aux différentes features x stockées dans X 
        per_obj[oid][0] = scaler.transform(per_obj[oid][0]).astype(np.float32)
    return scaler

## Dataset est une classe de base quasi-vide
# on doit redéfinir : le constructeur __init, la méthode __len, la méthode __getitem 
# en les implémentant, l'objet windowdataset se comporte comme un objet indexable

class WindowDataset(Dataset):
    """Séries (X, Y) par objet -> échantillons (fenêtre (F,W), cible Y[t])"""
    def __init__(self, per_obj, objects_ids, history=48, future=48):
        self.windowsize = history + future + 1 
        self.padded = {} # oid -> (Xpad, Y) pour bien gérer la fenètre glissante aux bords. Xpad est la matrice paddée des features .
        self.index = [] # index plat
        for oid in objects_ids:
            X, Y = per_obj[oid]
            Xpad = np.pad(X, ((history, future), (0,0)), mode='constant')
            self.padded[oid] = (Xpad, Y)
            self.index += [(oid, t) for t in range(len(X))]
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        oid, t = self.index[index]
        Xpad, Y = self.padded[oid]
        window = Xpad[t:t+ self.windowsize] # (W, Features)
        x = torch.from_numpy(window.T).float() # (Features, Window)
        y = torch.from_numpy(Y[t]).float() # (2, )
        return x,y

def make_loaders(objects, labels, batch_size=256, history=48, future=48, val_split=0.2, seed=42, half_width=6, **feature_kwargs):
    # arrays bruts par objet
    per_obj , feature_cols = build_arrays(objects, labels, half_width=half_width, **feature_kwargs)

    # split par objet 
    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    # scaler ajusté sur le train uniquement 
    scaler = fit_scaler_on_train(per_obj, train_ids)

    #Datasets + dataloader
    train_dataset = WindowDataset(per_obj, train_ids, history, future)
    val_dataset = WindowDataset(per_obj, val_ids,history, future)
    
    train_dl = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True
                          )
    
    val_dl =DataLoader(val_dataset, 
                        batch_size=batch_size,
                        shuffle=False
                        )
    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
            "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj}
    return train_dl, val_dl, meta

    
