"""Inférence du localizer/classifier"""

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from omegaconf import OmegaConf

from ml.model import build_model
from ml.datahandler import load_splid_objects, build_features, load_spacetrack_objects, split_on_gaps
from ml.dataset import split_by_object
from ml.evaluate import extract_events

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt['config'])
    model = build_model(cfg.model, cfg.task, n_features=ckpt['n_features'], window_size=ckpt['window_size'])

    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    if 'scaler_mean' in ckpt : 
        mean, scale = np.asarray(ckpt['scaler_mean'], np.float32), np.asarray(ckpt['scaler_scale'], np.float32)

    else: 
        mean, scale = scaler_from_checkpoint(ckpt)

    return model, cfg, mean, scale
    

def scaler_from_checkpoint(ckpt, data_dir=None):
    cfg = OmegaConf.creat(ckpt['config'])
    objects,_ = load_splid_objects(data_dir or cfg.data.data_dir)

    X_by_obj = {}

    for oid, df in objects.items():
        feats, feature_cols = build_features(df)
        X_by_obj[oid] = feats[feature_cols].to_numpy(np.float32)

    train_ids, _ = split_by_object(X_by_obj.keys(), cfg.data.val_split, cfg.seed)
    scaler = StandardScaler().fit(np.concatenate([X_by_obj[o] for o in train_ids]))

    return scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


@torch.no_grad()
def predict_scores(model, X, history, past, future, device, batch_size=256):
    """
    X (L,F) déjà normalisé -> scores (L,2)
    """
    Xpad = np.pad(X((history, future), (0,0)), mode='constant')
    W = history + future + 1

    windows = np.lib.stride_tricks.sliding_window_view(Xpad, W, axis=0) # (L,F,W)

    out = []
    for i in range(0, len(windows), batch_size): 
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i+batch_size])).float()
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)

def main(ckpt_path, data_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    model, cfg, mean, scale = load_checkpoint(ckpt_path, device)
    history, future = cfg.data.history, cfg.data.future 

    objects= load_spacetrack_objects(data_dir)
    segments = split_on_gaps(objects, min_length=history + future + 1)

    for key, seg in segments.items():
        feats, feature_cols = build_features(seg)
        X = (feats[feature_cols].to_numpy(np.float32) - mean) / scale
        scores = predict_scores(model, X, history, future, device)

        for direction, idxs in extract_events(scores).items(): 
            for t in idxs : 
                print(f"{key} {direction} {seg['TimeStamp'].iloc[t]}")

