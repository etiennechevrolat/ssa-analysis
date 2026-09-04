from ml.targets import (build_target, build_classifier_samples, build_doris_targets,
                        half_width_in_indices)
from ml.evaluate import doris_channels, gt_events_doris
from ml.datahandler import build_features
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


def _same_feature_cols(cols, feature_cols, oid):
    """Fige la liste de features au premier objet, puis vérifie que les suivants ont la MÊME.

    Les builders réassignaient `feature_cols` à chaque tour de boucle : la valeur qui finissait
    dans `meta` était celle du DERNIER objet traité — y compris un objet écarté juste après.
    Or cette liste va dans le checkpoint, sert d'indice au RevIN par fenêtre et de référence à
    check_backbone_compatibility : elle ne doit pas dépendre de l'ordre d'itération du dict.
    """
    cols = list(cols)
    if feature_cols is None:
        return cols
    if cols != feature_cols:
        raise ValueError(f"norad {oid} : features {cols}\n"
                         f"differentes des objets precedents : {feature_cols}")
    return feature_cols


def build_arrays(objects, labels=None, half_width=6):
    """
    chaque objet -> (X: (L,F), Y: (L,2))
    retourne per_obj, feature_cols
    """
    per_obj, feature_cols = {}, None
    for oid, df in objects.items():
        df_feat, cols = build_features(df, spacetrack=False)
        feature_cols = _same_feature_cols(cols, feature_cols, oid)
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
        df_feat, cols = build_features(df, spacetrack=False)
        feature_cols = _same_feature_cols(cols, feature_cols, oid)
        X = df_feat[feature_cols].to_numpy(np.float32)
        Y = build_classifier_samples(oid, labels)
        per_obj[oid] = [X,Y]
    return per_obj, feature_cols

def build_finetuning_arrays(objects, labels, half_width_hours=48.0,
                            detection_only=False, min_tle=0, min_maneuvers=1):
    """
    chaque objet -> (X: (L,F), Y: (L,2)) pour le finetuning sur les manoeuvres DORIS.
    features spacetrack (cadence irrégulière -> dt, sma en km) pour rester aligné sur le pretrain MAE.

    detection_only=True fusionne les deux types en un seul canal (L,1) : toute la supervision
    se concentre sur la tâche difficile, détecter un évènement, au lieu d'être partagée entre
    l'in-track et le cross-track, bien plus rare.

    min_tle / min_maneuvers : planchers sous lesquels un objet est écarté. Un objet plus court
    qu'une fenêtre ne produit que des fenêtres remplies de padding, et un objet à une ou deux
    manoeuvres pèse surtout par le bruit qu'il ajoute à la validation.
    """
    per_obj, feature_cols = {}, None
    retenus, ecartes = {}, {}
    for oid, df in objects.items():
        df_feat, cols = build_features(df, spacetrack=True)
        feature_cols = _same_feature_cols(cols, feature_cols, oid)
        X = df_feat[feature_cols].to_numpy(np.float32)
        Y = build_doris_targets(df, oid, labels, half_width_hours=half_width_hours)
        if detection_only:
            Y = Y.max(axis=1, keepdims=True) ## (L,1) : manoeuvre, tout type confondu

        ## Nombre de manoeuvres exploitables : on reprend gt_events_doris, donc EXACTEMENT le
        ## filtre de l'évaluation. Compter autrement ici reviendrait à écarter des objets sur
        ## un critère que la métrique ne partage pas.
        events = gt_events_doris(labels, [oid], detection_only=detection_only)[oid]
        n_maneuvers = sum(len(v) for v in events.values())

        if not Y.any(): ## série sans aucune manoeuvre utilisable
            ecartes[oid] = "cible entièrement nulle"
        elif n_maneuvers < min_maneuvers:
            ecartes[oid] = f"{n_maneuvers} manoeuvre(s) exploitable(s) < {min_maneuvers}"
        elif len(X) < min_tle:
            ecartes[oid] = f"série de {len(X)} TLE < {min_tle} (fenêtre entièrement paddée)"
        else:
            per_obj[oid] = [X, Y]
            retenus[oid] = (len(X), n_maneuvers)
            continue
        print(f"[build_finetuning_arrays] norad {oid} écarté : {ecartes[oid]}")

    ## < 2 objets : le split par objet n'a plus de sens, le seul partage possible laisse un
    ## train vide et la boucle d'entraînement diviserait par zéro sans dire pourquoi.
    if len(per_obj) < 2:
        raise ValueError(f"{len(per_obj)} objet(s) passent les planchers min_tle={min_tle}, "
                         f"min_maneuvers={min_maneuvers} : il en faut au moins 2 pour un split "
                         f"par objet (un en validation, un en entraînement)")

    print(f"[build_finetuning_arrays] {len(per_obj)} objets retenus sur {len(objects)} "
          f"(planchers : {min_tle} TLE, {min_maneuvers} manoeuvre(s))")
    for oid, (n_tle, n_man) in sorted(retenus.items(), key=lambda kv: kv[1][1]):
        print(f"    norad {oid} : {n_tle} TLE, {n_man} manoeuvres")
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

def fit_scaler_on_train(per_obj, train_ids, scaler=None, per_object=False):
    """Ajuste le scaler sur le train seulement, ou applique un scaler déjà ajusté.

    per_object=True : normalisation par objet (RevIN amont), scaler est un dict
    {norad: StandardScaler}. Rien ne se transfère du pretrain dans ce mode — mu et sigma ne
    sont pas des paramètres appris mais des statistiques de la série de l'objet
    On les recalcule donc sur sa propre série
    """
    if per_object:
        if scaler is not None:
            raise ValueError("per_object=True et scaler fourni : en normalisation par objet il "
                             "n'y a pas de scaler à réutiliser, mu/sigma se recalculent par objet")
        scaler = {}
        for oid in per_obj:
            scaler[oid] = StandardScaler().fit(per_obj[oid][0])
            per_obj[oid][0] = scaler[oid].transform(per_obj[oid][0]).astype(np.float32)
        return scaler

    if scaler is None:
        ## fitter le scaler = trouver nu et sigma correspondant aux valeurs des features dans train_ids
        scaler=StandardScaler().fit(np.concatenate([per_obj[o][0] for o in train_ids]))
    for oid in per_obj:
        ## applique x = (x-nu)/sigma aux différentes features x stockées dans X
        per_obj[oid][0] = scaler.transform(per_obj[oid][0]).astype(np.float32)
    return scaler

def fit_scaler_on_train_RevIn(per_obj, train_ids):
    """Ajuste un scaler standard sur chaque objet : scaler est un dict {norads : StandardScaler}"""
    return fit_scaler_on_train(per_obj, train_ids, per_object=True)

def scaler_from_checkpoint(ckpt, n_features):
    """Reconstruit le StandardScaler du pretrain depuis les champs sauvés par save_checkpoint."""
    mean, scale = ckpt.get('scaler_mean'), ckpt.get('scaler_scale')
    if mean is None or scale is None:
        raise ValueError("checkpoint sans scaler_mean/scaler_scale : impossible de réutiliser "
                         "la normalisation du pretrain, et un encodeur gelé l'exige")
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if mean.shape != (n_features,):
        raise ValueError(f"scaler du pretrain sur {mean.shape[0]} features, "
                         f"finetuning sur {n_features} : les jeux de features diffèrent")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = mean, scale, scale ** 2
    scaler.n_features_in_ = n_features
    return scaler

## Dataset est une classe de base quasi-vide
# on doit redéfinir : le constructeur __init, la méthode __len, la méthode __getitem 
# en les implémentant, l'objet windowdataset se comporte comme un objet indexable

class WindowDataset(Dataset):
    """Séries (X, Y) par objet -> échantillons (fenêtre (F,W), cible Y[t])

    pad_mode : padding des bords de série. 'constant' (zéros) place la valeur moyenne de
    l'objet — après centrage-réduction — sur toute la partie manquante, ce qui crée un saut
    artificiel. 'edge' prolonge la dernière valeur observée : indispensable dès que le modèle
    normalise par fenêtre (mu et sigma sont alors calculés sur ce padding), car le pretrain,
    lui, ne padde jamais (UnlabeledWindowDataset ne produit que des fenêtres pleines).
    """
    def __init__(self, per_obj, objects_ids, history=48, future=48, flatten_target=False,
                 pad_mode='constant'):
        self.windowsize = history + future + 1
        self.flatten_target = flatten_target # cible DORIS (L,2) -> (2,) pour une tête linéaire
        self.padded = {} # oid -> (Xpad, Y) pour bien gérer la fenètre glissante aux bords. Xpad est la matrice paddée des features .
        self.index = [] # index plat
        for oid in objects_ids:
            X, Y = per_obj[oid]
            Xpad = np.pad(X, ((history, future), (0,0)), mode=pad_mode)
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
        y = torch.from_numpy(np.ascontiguousarray(target)).float() # (2, ) : EW/NS pour SPLID, in-track/cross-track pour DORIS
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
                            val_split=0.2, seed=42, half_width_hours=48.0, flatten_target=True,
                            detection_only=False, scaler=None, per_object_scaler=False,
                            tolerance_hours=48.0, pad_mode='edge',
                            min_tle=None, min_maneuvers=1):
    """
    Loaders pour le finetuning DORIS. Cible (L,2) aplatie en (2,) par défaut, (1,) si
    detection_only.
    Split simple groupé par objet (val_split) : avec les ~200 objets de l'annotation SSO, la
    validation en compte assez pour que le chiffre ne dépende plus du tirage — ce que le
    leave-one-out compensait quand le jeu DORIS n'avait que 13 objets exploitables.
    scaler non None -> on réutilise celui du pretrain au lieu d'en ajuster un nouveau.
    per_object_scaler=True -> normalisation par objet, recalculée sur chaque série DORIS
    (cf. fit_scaler_on_train) ; c'est le mode des checkpoints scaler_kind='per_obj'.
    pad_mode='edge' par défaut ici : cf. WindowDataset, un backbone qui normalise par fenêtre
    calculerait sinon mu et sigma sur des zéros de padding.
    """
    ## min_tle=None -> une fenêtre complète : en dessous, l'objet n'a pas une seule fenêtre
    ## sans padding, et le pretrain n'en a jamais vu de telles.
    if min_tle is None:
        min_tle = history + future + 1
    per_obj, feature_cols = build_finetuning_arrays(objects, labels, half_width_hours=half_width_hours,
                                                    detection_only=detection_only,
                                                    min_tle=min_tle, min_maneuvers=min_maneuvers)

    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    scaler = fit_scaler_on_train(per_obj, train_ids, scaler=scaler, per_object=per_object_scaler)
    
    train_dataset = WindowDataset(per_obj, train_ids, history, future,
                                  flatten_target=flatten_target, pad_mode=pad_mode)
    val_dataset = WindowDataset(per_obj, val_ids, history, future,
                                flatten_target=flatten_target, pad_mode=pad_mode)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, persistent_workers=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)

    target_shape = next(iter(per_obj.values()))[1].shape[1:] ## (2,) ou (1,) si detection_only

    ## tolérance d'appariement en HEURES -> indices, par objet (cf. half_width_hours)
    tolerance = {oid : half_width_in_indices(objects[oid], tolerance_hours) for oid in per_obj}

    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
            "scaler_kind" : 'per_obj' if per_object_scaler else 'global',
            "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj,
            "half_width_hours" : half_width_hours,
            "detection_only" : detection_only,
            "min_tle" : min_tle, "min_maneuvers" : min_maneuvers,
            "channels" : ('maneuver',) if detection_only else doris_channels,
            "tolerance" : tolerance, "tolerance_hours" : tolerance_hours,
            "target_shape" : target_shape,
            "n_outputs" : int(np.prod(target_shape)) if flatten_target else target_shape}
    return train_dl, val_dl, meta

def make_pretrain_loader(objects, window_size, stride, batch_size=256, val_split=0.2, seed=42, revin= False):
    per_obj, feature_cols =  {}, None
    for oid, df in objects.items() :
        df_feat, cols = build_features(df, spacetrack=True)
        feature_cols = _same_feature_cols(cols, feature_cols, oid)
        per_obj[oid] = [df_feat[feature_cols].to_numpy(np.float32), None]
    if feature_cols is None:
        raise ValueError("aucun objet chargé : feature_cols indéterminé")

    train_ids, val_ids = split_by_object(per_obj.keys(), val_split, seed)

    
    if revin : 
        ## les données sont normalisées par objet 
        scaler = fit_scaler_on_train_RevIn(per_obj, train_ids)
    else : 
        scaler = fit_scaler_on_train(per_obj, train_ids)

    train_dataset = UnlabeledWindowDataset(per_obj, train_ids, window_size, stride)
    val_dataset = UnlabeledWindowDataset(per_obj, val_ids, window_size, stride)

    ## drop_last : sans lui le dernier batch de chaque epoch est partiel, donc d'une forme
    ## inedite, et torch.compile recompile le graphe entier (~7 min) a chaque fois. On perd
    ## au plus 255 fenetres sur ~770 000, et la val loss reste comparable d'une epoch a l'autre
    ## puisqu'on ecarte toujours le meme reliquat.
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True,
                        persistent_workers=True, drop_last=True)
    meta = {"feature_cols" : feature_cols, "scaler" : scaler,
                "train_ids" : train_ids, "val_ids" : val_ids, "per_obj" : per_obj}
    return train_dl, val_dl, meta