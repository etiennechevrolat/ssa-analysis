from ml.targets import build_target, build_classifier_samples, build_doris_targets, DELTA_V_THRESHOLD
from ml.datahandler import build_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


def build_arrays(objects, labels=None, half_width=6):
    """
    chaque objet -> (X: (L,F), Y: (L,2))
    retourne per_obj, feature_cols
    """
    per_obj, feature_cols = {}, None
    for oid, df in objects.items():
        df_feat, feature_cols = build_features(df, spacetrack=False)
        X = df_feat[feature_cols].to_numpy(np.float32)
        if labels is not None :
            Y = build_target(df, oid, labels, half_width=half_width)
        else :
            Y = None
        per_obj[oid] = [X,Y]
    return per_obj, feature_cols

def build_classifier_arrays(objects,labels):
    """
    chaque objet -> (X:(L,F), Y: node samples créée par targets.py)
    Renvoie per_obj, feature_cols équivalent de build-arrays pour classification
    """
    per_obj, feature_cols = {}, None
    for oid, df in objects.items(): ## df est le dataframe brut de donnée
        df_feat, feature_cols = build_features(df, spacetrack=False)
        X = df_feat[feature_cols].to_numpy(np.float32)
        Y = build_classifier_samples(oid, labels)
        per_obj[oid] = [X,Y]
    return per_obj, feature_cols

def build_finetuning_arrays(objects, labels, half_width=6, thresholds=DELTA_V_THRESHOLD):
    """
    chaque objet -> (X: (L,F), Y: (L,2,2)) pour le finetuning sur les manoeuvres DORIS.
    features spacetrack (cadence irrégulière -> dt, sma en km) pour rester aligné sur le pretrain MAE.
    """
    per_obj, feature_cols = {}, None
    for oid, df in objects.items():
        df_feat, feature_cols = build_features(df, spacetrack=True)
        X = df_feat[feature_cols].to_numpy(np.float32)
        Y = build_doris_targets(df, oid, labels, half_width=half_width, thresholds=thresholds)
        if not Y.any(): ## série sans aucune manoeuvre utilisable :
            print(f"[build_finetuning_arrays] norad {oid}: cible entièrement nulle, objet écarté")
            continue
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

    ## .tolist() sur un tableau d'entiers restitue des int python (les clés des dicts d'objets)
    return(ids[n_val:].tolist(), ids[:n_val].tolist())

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
    def __init__(self, per_obj, objects_ids, history=48, future=48, flatten_target=False):
        self.windowsize = history + future + 1
        self.flatten_target = flatten_target # cible DORIS (L,2,2) -> (4,) pour une tête linéaire
        self.padded = {} # oid -> (Xpad, Y) pour bien gérer la fenètre glissante aux bords. Xpad est la matrice paddée des features .
        self.index = [] # index plat
        for oid in objects_ids:
            X, Y = per_obj[oid]
            Xpad = np.pad(X, ((history, future), (0,0)), mode='constant')
            self.padded[oid] = (Xpad, Y)
            self.index += [(oid, t) for t in range(len(X))] ### len(X) = nombre de pas de temps
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        oid, t = self.index[index] ## on récupére le bon objet sachant qu'on a une indexation absolue et que l'objet oid possède t features ?
        Xpad, Y = self.padded[oid]
        window = Xpad[t:t+ self.windowsize] # (W, Features)
        x = torch.from_numpy(window.T).float() # (Features, Window)  + transformation en tenseurs pytorch
        target = Y[t].reshape(-1) if self.flatten_target else Y[t]
        y = torch.from_numpy(np.ascontiguousarray(target)).float() # (2, ) ou (4, ) pour DORIS (qui prend en compte l'intensité de la manoeuvre)
        return x,y


class NodeWindowDataset(Dataset):
    """Séries (X, Y) par objet -> échantillons (fenêtre (F,W), cible Y[t])
    Dans ce cas, self.index ne contient plus tous les t, mais seulement les noeuds
    """
    def __init__(self, per_obj, objects_ids, history=48, future=48):
        self.windowsize = history + future + 1 
        self.padded = {} # oid -> (Xpad, Y) pour bien gérer la fenètre glissante aux bords. Xpad est la matrice paddée des features .
        self.index = [] # index plat
        for oid in objects_ids:
            X, node_samples = per_obj[oid]
            Xpad = np.pad(X, ((history, future), (0,0)), mode='constant')
            self.padded[oid] = (Xpad, node_samples)
            self.index += [(oid, time_index, dir, node_type, class_type) for time_index, dir, node_type, class_type in node_samples]

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        oid, t, dir, node_type, class_type = self.index[index]  
        Xpad, node_samples = self.padded[oid]
        window = Xpad[t:t+ self.windowsize] # (W, Features) 

        x = torch.from_numpy(window.T).float() # (Features, Window)  + transformation en tenseurs pytorch 
        y = {
            'node' : torch.tensor(node_type, dtype=torch.long),
            'class' : torch.tensor(class_type, dtype=torch.long)
            }

        return x,y

class UnlabeledWindowDataset(Dataset):
    """Séries longues par objet -> fenêtres (F,W) sans label, pour le preentrainement MAE"""

    def __init__(self, per_obj, objects_ids, window_size, stride):
        self.series = {}
        self.index = [] # index plat
        self.window_size = window_size
        for oid in objects_ids:
            X, _= per_obj[oid]
            if len(X) < window_size: 
                continue
            self.series[oid] = X

            self.index += [(oid, t) for t in range(0, len(X) - window_size + 1, stride)] ### len(X) = nombre de pas de temps
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        oid, t = self.index[index]
        window = self.series[oid][t:t+ self.window_size] # (W, F) 
        x = torch.from_numpy(window.T).float() # (Features, Window)  + transformation en tenseurs pytorch 
        return x 

## Les dataloader pour le training
def make_loaders(objects, labels, batch_size=256, history=48, future=48, val_split=0.2, seed=42, half_width=6):
    # arrays bruts par objet
    per_obj , feature_cols = build_arrays(objects, labels, half_width=half_width)
    
    # split par objet 
    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    # scaler ajusté sur le train uniquement 
    scaler = fit_scaler_on_train(per_obj, train_ids)

    #Datasets + dataloader
    train_dataset = WindowDataset(per_obj, train_ids, history, future)
    val_dataset = WindowDataset(per_obj, val_ids,history, future)

    train_dl = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          persistent_workers=True
                          )

    val_dl =DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        persistent_workers=True
                        )
    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
            "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj}
    return train_dl, val_dl, meta

def make_loaders_classifiers(objects, labels, batch_size=256, history=48, future=48, val_split=0.2, seed=42):
    # arrays bruts par objet
    per_obj , feature_cols = build_classifier_arrays(objects, labels)
    
    # split par objet 
    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    # scaler ajusté sur le train uniquement 
    scaler = fit_scaler_on_train(per_obj, train_ids)

    #Datasets + dataloader
    train_dataset = NodeWindowDataset(per_obj, train_ids, history, future)
    val_dataset = NodeWindowDataset(per_obj, val_ids,history, future)

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

def make_loaders_finetuning(objects, labels, batch_size=256, history=48, future=47,
                            val_split=0.2, seed=42, half_width=6, flatten_target=True,
                            thresholds=DELTA_V_THRESHOLD):
    """
    Loaders pour le finetuning DORIS. Cible (L,2,2) aplatie en (4,) par défaut.
    Attention : peu d'objets (15 norads au plus), donc val_ids est très petit et non stratifié
    par type de manoeuvre -> possible de tirer un val sans aucune cross-track.
    """
    per_obj, feature_cols = build_finetuning_arrays(objects, labels, half_width=half_width,
                                                    thresholds=thresholds)

    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    scaler = fit_scaler_on_train(per_obj, train_ids)

    train_dataset = WindowDataset(per_obj, train_ids, history, future, flatten_target=flatten_target)
    val_dataset = WindowDataset(per_obj, val_ids, history, future, flatten_target=flatten_target)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)

    target_shape = next(iter(per_obj.values()))[1].shape[1:] ## (2, C)
    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
            "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj,
            "delta_v_thresholds" : thresholds, "half_width" : half_width,
            "target_shape" : target_shape,
            "n_outputs" : int(np.prod(target_shape)) if flatten_target else target_shape}
    return train_dl, val_dl, meta

def make_pretrain_loader(objects, window_size, stride, batch_size=256, val_split=0.2, seed=42):
    per_obj =  {}
    for oid, df in objects.items() :
        df_feat, feature_cols = build_features(df, spacetrack=True)
        per_obj[oid] = [df_feat[feature_cols].to_numpy(np.float32), None]

    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    scaler = fit_scaler_on_train(per_obj, train_ids)

    train_dataset = UnlabeledWindowDataset(per_obj, train_ids, window_size, stride)
    val_dataset = UnlabeledWindowDataset(per_obj, val_ids, window_size, stride)

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
                "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj}
    return train_dl, val_dl, meta