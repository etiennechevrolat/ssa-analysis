"""Inférence.

Deux usages :
- localizer/classifier supervisé (cf. main()).
  Premiers résultats sur SpaceTrack : extrémement mauvais. voué à l'échec : tle spacetrack sortent des paramètres moyennés, à intervalle de temps irréguliers.
  Les données Splid sont très limitées : du GEO, quasi equatorial.
- backbone MAE préentrainé : extraction des représentations (token CLS) puis
  clusterisation HDBSCAN du catalogue LEO (cf. src/viz/clustering_leo.ipynb).
"""

from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.model import build_model, VanillaViT
from ml.datahandler import load_splid_objects, load_spacetrack_objects, build_features, load_spacetrack_objects_to_splid, split_on_gaps
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
    cfg = OmegaConf.create(ckpt['config'])
    objects,_ = load_splid_objects(data_dir or cfg.data.data_dir)

    X_by_obj = {}

    for oid, df in objects.items():
        feats, feature_cols = build_features(df, spacetrack=False)
        X_by_obj[oid] = feats[feature_cols].to_numpy(np.float32)

    train_ids, _ = split_by_object(X_by_obj.keys(), cfg.data.val_split, cfg.seed)
    scaler = StandardScaler().fit(np.concatenate([X_by_obj[o] for o in train_ids]))

    return scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


@torch.no_grad()
def predict_scores(model, X, history, future, device, batch_size=256):
    """
    X (L,F) déjà normalisé -> scores (L,2)
    """
    Xpad = np.pad(X, ((history, future), (0,0)), mode='constant')
    W = history + future + 1

    windows = np.lib.stride_tricks.sliding_window_view(Xpad, W, axis=0) # (L,F,W)

    out = []
    for i in range(0, len(windows), batch_size): 
        x = torch.from_numpy(np.ascontiguousarray(windows[i:i+batch_size])).float()
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)

## Backbone MAE préentrainé : extraction des représentations

def load_pretrained_backbone(ckpt_path, device):
    """Charge un checkpoint de pretraining (MAEv1/MAEv2/miniMAE) et transfère les
    poids de l'encodeur dans un VanillaViT de même architecture

    Retourne (backbone, cfg, mean, scale) : mean/scale sont ceux du scaler de pretraining.
    """
    mae, cfg, mean, scale = load_checkpoint(ckpt_path, device)

    backbone = VanillaViT(
        n_features=mae.patch_embedding.n_features,
        n_epochs=mae.patch_embedding.window_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.encoder_embed_dim,
        n_attn_heads=cfg.model.encoder_n_attn_heads,
        n_blocks=cfg.model.encoder_n_blocks,
        expansion_factor=cfg.model.expansion_factor,
        dropout_rate=cfg.model.dropout_rate,
    )

    ## encoder_state_dict() donne directement les poids de l'encodeur avec les bons mappings
    missing, unexpected = backbone.load_state_dict(mae.encoder_state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"poids inattendus dans l'encodeur : {unexpected}")

    return backbone.to(device).eval(), cfg, mean, scale


## Diagnostic du pretraining : le MAE reconstruit-il les manoeuvres, ou les lisse-t-il ?

@torch.no_grad()
def reconstruct_window(mae, x, masked_patches=None, device='cpu', seed=0):
    """Reconstruction COMPLETE d'une fenêtre par le MAE, avec contrôle du masque.

    x : (F, W) normalisé.
    masked_patches : indices des patchs à masquer. None -> tirage aléatoire au taux de
        masquage du pretraining. Passer explicitement le patch qui contient une manoeuvre
        permet de voir si le décodeur restitue le saut ou le remplace par une interpolation.

    Retourne (reconstruction (F,W) normalisée, masked_patches (array)).

    Reproduit TimeSeriesMAE.forward, mais renvoie TOUS les patchs reconstruits : le forward
    ne renvoie que les patchs masqués, ce qui suffit à la loss mais pas à tracer une série.
    """
    mae = mae.to(device).eval()
    n_features, window_size = x.shape
    patch_size = mae.patch_embedding.patch_size
    N = mae.num_patches

    if masked_patches is None:
        rng = np.random.default_rng(seed)
        n_mask = round(N * mae.masking_ratio)
        masked_patches = np.sort(rng.choice(N, size=n_mask, replace=False))
    masked_patches = np.atleast_1d(np.asarray(masked_patches, dtype=int))
    kept_patches = np.setdiff1d(np.arange(N), masked_patches)

    ## ids_shuffle = [gardés, masqués] : même convention que le forward, où l'encodeur ne
    ## voit que le début de la permutation et ids_restore remet l'ordre chronologique.
    ids_shuffle = torch.tensor(np.concatenate([kept_patches, masked_patches]), device=device)[None]
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    xb = torch.from_numpy(np.ascontiguousarray(x)).float()[None].to(device) # (1,F,W)
    tokens = mae.encoder_pos_embedding(mae.patch_embedding(xb))             # (1,N,D_enc)
    embed_dim = tokens.shape[-1]

    visible = torch.gather(tokens, 1, ids_shuffle[:, :len(kept_patches), None].expand(-1, -1, embed_dim))
    h = torch.cat([mae.cls_token.expand(1, -1, -1), visible], dim=1)
    for bloc in mae.encoder_blocks:
        h = bloc(h)
    h = mae.encoder_norm(h)

    y = mae.encoder_to_decoder_embedding(h)
    decoder_dim = y.shape[-1]
    y_ = torch.cat([y[:, 1:], mae.mask_token.expand(1, len(masked_patches), -1)], dim=1)
    y_ = torch.gather(y_, 1, ids_restore[..., None].expand(-1, -1, decoder_dim))

    y = mae.decoder_pos_embedding(torch.cat([y[:, :1], y_], dim=1))
    for bloc in mae.decoder_blocks:
        y = bloc(y)
    y = mae.decoder_norm(y)

    pred = mae.pred_space_proj(y[:, 1:])                        # (1,N,F*P)
    ## inverse exact du target_all du forward : (1,N,F*P) -> (1,N,F,P) -> (1,F,N,P) -> (1,F,W)
    pred = pred.reshape(1, N, n_features, patch_size).permute(0, 2, 1, 3).reshape(1, n_features, window_size)

    return pred[0].cpu().numpy(), masked_patches


def patch_of_index(time_index, window_start, patch_size):
    """Indice du patch contenant time_index, dans une fenêtre commençant à window_start."""
    return (time_index - window_start) // patch_size


def load_spacetrack_features(data_dir, dataset, mean, scale):
    """Features SpaceTrack normalisées avec le scaler du pretraining : {norad: (L,F)}"""
    per_obj = {}
    for norad, df in load_spacetrack_objects(data_dir, dataset).items():
        features, feature_cols = build_features(df, spacetrack=True)
        per_obj[norad] = (features[feature_cols].to_numpy(np.float32) - mean) / scale
    return per_obj


@torch.no_grad()
def embed_objects(backbone, per_obj, device, window_size=96, stride=13, batch_size=256):
    """Une représentation par objet : token CLS (indice 0) moyenné sur toutes les fenêtres.

    per_obj : {norad: (L,F)} déjà normalisé. Les objets plus courts qu'une fenêtre sont ignorés.
    Retourne {norad: (embed_dim,)}
    """
    backbone.to(device).eval()

    out = {}
    for norad, X in per_obj.items():
        if len(X) < window_size:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(X, window_size, axis=0)[::stride]  # (L,F,W)

        cls_representations = []
        for i in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(windows[i:i + batch_size])).float().to(device)
            representation = backbone(x)
            cls_representations.append(representation[:, 0, :].cpu().numpy())  ## on ne stocke que le cls token

        if cls_representations:
            out[norad] = np.concatenate(cls_representations, axis=0).mean(axis=0)
    return out

## Clusterisation HDBSCAN des représentations

def cluster_embeddings(embeddings, min_cluster_size=15, min_samples=15):
    """HDBSCAN sur les représentations standardisées.

    embeddings : {norad: (D,)}
    Retourne (norad_ids, X_scaled, labels) ; label -1 = bruit.
    """
    norad_ids = list(embeddings.keys())
    X_scaled = StandardScaler().fit_transform(np.stack([embeddings[n] for n in norad_ids]))

    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X_scaled)

    return X_scaled, labels, norad_ids


def drop_noise(X, labels, norad_ids=None):
    """Retire les objets non clusterisés (label -1)."""
    mask = labels != -1
    if norad_ids is None:
        return X[mask], labels[mask]
    return X[mask], labels[mask], [n for n, keep in zip(norad_ids, mask) if keep]


def tune_hdbscan(X, cluster_sizes=range(2, 21, 3), sample_counts=range(2, 21, 3)):
    """Balayage des paramètres HDBSCAN, sélection au DBCV (relative_validity_).

    Retourne (best_min_cluster_size, best_min_samples, best_score).
    """
    best = (0, 0, -np.inf)
    for min_cluster_size in cluster_sizes:
        for min_samples in sample_counts:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
            clusterer.fit(X)
            if clusterer.relative_validity_ > best[2]:
                best = (min_cluster_size, min_samples, clusterer.relative_validity_)
    return best


def cluster_train_val(embeddings, min_cluster_size=15, min_samples=15, val_split=0.2, seed=42):
    """Clusterise séparément le train et la val (même split que le pretraining) pour
    vérifier que le backbone généralise : les deux nuages doivent avoir la même structure.

    Retourne {'train': (norad_ids, X_scaled, labels), 'val': (...)}.
    """
    train_ids, val_ids = split_by_object(object_ids=list(embeddings.keys()), val_split=val_split, seed=seed)

    return {
        split: cluster_embeddings({n: embeddings[n] for n in ids}, min_cluster_size, min_samples)
        for split, ids in (('train', train_ids), ('val', val_ids))
    }


def save_clusters_csv(norad_ids, labels, out_path):
    """Sauve l'appartenance norad -> cluster."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'norad': norad_ids, 'cluster': labels}).to_csv(out_path, index=False)
    return out_path

## Visualisation des clusters


def _clusters_title(cfg, labels, min_cluster_size, min_samples):
    return (f"HDBSCAN Clustering - {len(set(labels))} clusters, {len(labels)} objets"
            f" - min_cluster_size = {min_cluster_size}, min_samples = {min_samples} - "
            f"pretrain MAEv2 architecture - {cfg.model.embed_dim}D, {cfg.model.masking_ratio} mask, {cfg.data.window_size}Window size")


def plot_clusters_2d(cfg, X, labels, min_cluster_size=15, min_samples=15, out_path=None, show=True):
    """X (N,D) déjà standardisé : réduction à 2 composantes principales et nuage coloré par cluster.
    Chaque cluster est annoté à son barycentre.
    """
    X_2d = PCA(n_components=2, svd_solver='full').fit_transform(X) if X.shape[1] > 2 else X
    df = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1], 'cluster': labels.astype(str)})

    fig, ax = plt.subplots(figsize=(25, 18))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='husl', s=10, alpha=0.9, legend=False, ax=ax)

    for cluster_id, group in df.groupby('cluster'):
        ax.annotate(str(cluster_id), (group['x'].mean(), group['y'].mean()), fontsize=8, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black'), alpha=0.75)

    ax.set_title(_clusters_title(cfg, labels, min_cluster_size, min_samples))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    return _finish_plot(fig, out_path, show)


def plot_clusters_3d(X, labels, min_cluster_size=15, min_samples=15, out_path=None, show=True):
    """Même chose que plot_clusters_2d mais sur 3 composantes principales."""
    X_3d = PCA(n_components=3, svd_solver='full').fit_transform(X) if X.shape[1] > 3 else X
    df = pd.DataFrame({'x': X_3d[:, 0], 'y': X_3d[:, 1], 'z': X_3d[:, 2], 'cluster': labels.astype(str)})

    fig = plt.figure(figsize=(25, 18))
    ax = fig.add_subplot(projection='3d')

    palette = sns.color_palette('husl', df['cluster'].nunique())
    for color, (cluster_id, group) in zip(palette, df.groupby('cluster')):
        ax.scatter(group['x'], group['y'], group['z'], color=color, s=10, alpha=0.9)
        ax.text(group['x'].mean(), group['y'].mean(), group['z'].mean(), str(cluster_id),
                fontsize=8, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black'), alpha=0.75)

    ax.set_title(_clusters_title(labels, min_cluster_size, min_samples))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()

    return _finish_plot(fig, out_path, show)


def _finish_plot(fig, out_path, show):
    """Sauve la figure dans data/graphs (ou ailleurs) et/ou l'affiche dans le notebook."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main(ckpt_path, data_dir, out_dir):
    ## script de chargement des encodeurs préentrainés
    print("[1/5] Chargement du device...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    print("[2/5] Chargement du backbone préentraîné...")
    backbone, cfg, mean, scale = load_pretrained_backbone(ckpt_path, device)
    window_size = cfg.data.window_size
    stride = cfg.data.stride

    print("[3/5] Chargement et préparation des données...")
    per_obj = load_spacetrack_features(data_dir, 'leo', mean, scale)
    print("[4/5] Calcul des embeddings...")
    out = embed_objects(backbone, per_obj, device, window_size, stride)

    min_cluster_size, min_samples= 15,15
    print("[5/5] Clustering et génération du graphique...")
    x, labels, norads = cluster_embeddings(out, min_cluster_size, min_samples)
    x, labels, norads = drop_noise(x,labels, norads)

    plot_clusters_2d(cfg, x,labels, min_cluster_size, min_samples, out_path=out_dir)

if __name__ == "__main__" : 
    import os 
    from pathlib import Path

    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = Path(os.path.join(base, '..' , '..', 'outputs', 'ml', 'pretrain', '2026-08-11_14-43-36', 'checkpoints', 'best.pt' ))
    data_dir = Path(os.path.join(base, '..' , '..', 'data', 'raw', 'spacetrack'))
    out_dir = os.path.join(base, '..', '..', 'data', 'graphs', 'clustering')

    main(ckpt_path, data_dir, out_dir)


    