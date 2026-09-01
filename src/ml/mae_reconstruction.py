from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ml.inference import (load_checkpoint, load_spacetrack_features, reconstruct_window,
                          embed_objects, cluster_embeddings)


def load_mae_and_data(ckpt_id, base_path=None):
    """Charge le modele et les features normalisees 'dataset' (avant RevIN par fenetre).

    Deux normalisations distinctes et independantes entrent en jeu :
      - normalisation dataset (globale ou par objet), appliquee ICI sur per_obj. Son inverse
        est stats[norad] = (mu, sigma) : en mode global tous les objets partagent le meme
        (mu, sigma) ; en mode par objet (scaler_kind='per_obj') chacun a le sien. C'est
        load_spacetrack_features qui decide, via mean/scale issus du checkpoint : c'est
        la SEULE source de verite pour repasser en unites physiques, quel que soit le mode.
      - RevIN par fenetre, interne au modele (TimeSeriesMAE._instance_stats), applique ET
        annule entierement a l'interieur de reconstruct_window. L'appelant n'a jamais besoin
        d'y toucher : reconstruct_window renvoie deja la reconstruction dans l'espace de la
        normalisation dataset ci-dessus, prete a etre multipliee par stats[norad].
    """
    ## racine du repo deduite de l'emplacement de ce module (src/ml/ -> ../..), et non du
    ## cwd : un notebook demarre son kernel dans son propre dossier, un script lance depuis
    ## la racine non, et un Path.cwd() relatif casse dans l'un des deux cas.
    if base_path is None:
        base_path = Path(__file__).resolve().parents[2]
    base_path = Path(base_path).resolve()

    ckpt_path = base_path / "outputs" / "ml" / "pretrain" / ckpt_id / "checkpoints" / "best.pt"
    data_dir = base_path / "data" / "raw" / "spacetrack"

    mae, cfg, mean, scale = load_checkpoint(ckpt_path, device="cpu")

    per_obj, stats, feature_cols = load_spacetrack_features(
        data_dir, cfg.data.dataset, mean, scale, return_stats=True
    )
    per_obj = {norad: X for norad, X in per_obj.items() if len(X) >= cfg.data.window_size}
    stats = {norad: mu_sigma for norad, mu_sigma in stats.items() if norad in per_obj}

    return mae, cfg, stats, per_obj, feature_cols


def plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad, n_windows=3,
                        to_plot=("sma", "sma_diff"), save_path=None, show=True):
    """Trace n_windows fenetres d'un objet, original vs reconstruction sur patchs masques.

    save_path : si fourni, ecrit UNE figure regroupant les n_windows fenetres et la ferme
    au lieu de l'afficher -- indispensable pour un export en lot, sinon matplotlib garde
    toutes les figures ouvertes et sature la memoire.
    """

    window_size = cfg.data.window_size
    patch_size = cfg.model.patch_size

    to_plot = list(to_plot)
    X = per_obj[norad]
    starts = np.linspace(0, len(X) - window_size, n_windows).astype(int)
    device = next(mae.parameters()).device
    mu, sigma = stats[norad]

    n_lignes = len(starts) * len(to_plot)
    fig, axes = plt.subplots(n_lignes, 1, figsize=(14, 2.4 * n_lignes), squeeze=False)
    axes = axes[:, 0]

    for i, window_start in enumerate(starts):
        x = X[window_start : window_start + window_size].T
        recon, masked = reconstruct_window(mae, x, device=device, seed=0)

        ## mu, sigma : normalisation dataset de cet objet (globale ou par objet selon le
        ## checkpoint, cf. load_mae_and_data). recon est deja debarrasse du RevIN par fenetre.
        x_phys = x.T * sigma + mu
        recon_phys = recon.T * sigma + mu
        t = np.arange(window_start, window_start + window_size)

        for j, name in enumerate(to_plot):
            ax = axes[i * len(to_plot) + j]
            col = feature_cols.index(name)
            for p in masked:
                ax.axvspan(t[p * patch_size], t[min((p + 1) * patch_size, window_size - 1)],
                           color="grey", alpha=0.18, lw=0)

            pred = np.full(window_size, np.nan)
            for p in masked:
                pred[p * patch_size : (p + 1) * patch_size] = recon_phys[p * patch_size : (p + 1) * patch_size, col]

            ax.plot(t, x_phys[:, col], color="black", lw=1.3, label="original")
            ax.plot(t, pred, color="tab:red", lw=1.6, label="reconstruit")
            ax.set_ylabel(name)
            ax.margins(x=0)
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(f"norad {norad} — patchs masqués grisés")
    axes[-1].set_xlabel("TimeIndex (TLE)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    elif show:
        plt.show()
    return fig


## Repartition de la loss par canal

## Doit rester le miroir exact de la construction de w dans train.py (bloc is_pretrain) :
## w part de 1.0 partout, et seules les entrees listees ici sont ecrasees par la config.
## Un canal absent de la config garde donc 1.0, comme a l'entrainement.
def loss_weights(cfg, feature_cols):
    """Vecteur de ponderation par canal effectivement utilise par MaskedChannelMSE.

    Doit rester le miroir exact du bloc is_pretrain de train.py : w part de 1.0 partout,
    et seules les entrees de cfg.task.channel_weights sont ecrasees. Un canal absent de la
    config garde donc 1.0, comme a l'entrainement.
    """
    w = np.ones(len(feature_cols), dtype=np.float32)
    for canal, val in (cfg.task.get("channel_weights") or {}).items():
        if canal in feature_cols:
            w[list(feature_cols).index(canal)] = float(val)
    return w


@torch.no_grad()
def channel_loss_breakdown(mae, cfg, per_obj, feature_cols, n_windows=2048,
                           batch_size=256, seed=0, device='cpu'):
    """Part de la loss attribuable a chaque canal, sur un echantillon de fenetres.

    Reproduit MaskedChannelMSE : d2[f] est la MSE du canal f sur les patchs MASQUES
    uniquement, et loss = sum(d2 * w) / sum(w). D'ou :
      - 'contribution' = d2*w / sum(w) : les lignes somment exactement a la loss ;
      - 'part_%'       = d2*w / sum(d2*w) : poids relatif du canal dans ce que le
        modele optimise reellement. Un canal a poids 0 ressort a 0 % meme si sa MSE
        est enorme -- c'est precisement ce qu'on cherche a voir.

    La colonne 'mse' est elle NON ponderee : elle dit a quel point le canal est mal
    reconstruit, independamment du fait qu'on le penalise ou non.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    W = cfg.data.window_size
    F = len(feature_cols)

    eligibles = [n for n, X in per_obj.items() if len(X) >= W]
    if not eligibles:
        raise ValueError(f"aucun objet d'au moins {W} pas de temps")

    fenetres = []
    for _ in range(n_windows):
        X = per_obj[eligibles[rng.integers(len(eligibles))]]
        start = rng.integers(0, len(X) - W + 1)
        fenetres.append(X[start:start + W].T)          # (F, W)

    mae = mae.to(device).eval()
    somme_carres = torch.zeros(F, dtype=torch.float64)
    n_termes = 0
    for i in range(0, len(fenetres), batch_size):
        xb = torch.from_numpy(np.stack(fenetres[i:i + batch_size])).float().to(device)
        pred, target = mae(xb)
        B, N, _ = pred.shape
        d = (pred - target).reshape(B, N, F, -1)       # (B, N_masques, F, patch_size)
        somme_carres += d.pow(2).sum(dim=(0, 1, 3)).double().cpu()
        n_termes += B * N * d.shape[-1]

    d2 = (somme_carres / n_termes).numpy()
    w = loss_weights(cfg, feature_cols)
    contrib = d2 * w
    total = contrib.sum()

    return pd.DataFrame({
        'canal': list(feature_cols),
        'poids': w,
        'mse': d2,
        'contribution': contrib / w.sum(),
        'part_%': (contrib / total * 100) if total > 0 else np.zeros(F),
    }).sort_values('part_%', ascending=False).reset_index(drop=True)


## Export en lot des reconstructions

def maneuver_profile(sma, w=10, seuil=10.0):
    """Profil de manoeuvres d'un objet, a partir de sa serie de demi-grand axe.

    Une manoeuvre est un changement PERSISTANT de niveau : on compare la mediane des w
    points avant et des w points apres, et on retient les ecarts anormaux par rapport a la
    derive courante (le sma decroit en permanence, d'ou le centrage sur la mediane des sauts).

    Le comptage naif de pics sur sma_diff ne convient pas : il remonte les objets a jitter
    permanent (maintien a poste bruite) et, surtout, il note 1 seul "saut" pour une montee
    d'orbite Starlink qui est une rampe continue de 400 km. Ce qu'on cherche, ce sont
    PLUSIEURS sauts discrets d'amplitude comparable separes par de la decroissance libre.

    sma peut etre en unites normalisees : toutes les grandeurs renvoyees sont des rapports,
    donc invariantes d'echelle.
    """
    sma = np.asarray(sma, dtype=np.float64)
    n = len(sma)
    vide = {"n_sauts": 0, "saut_median_bruit": np.nan, "amplitude_bruit": np.nan}
    if n < 4 * w:
        return vide

    ## bruit haute frequence : la difference seconde annule une tendance lineaire locale,
    ## le MAD la rend insensible aux sauts eux-memes. /sqrt(6) car var(d2) = 6*var(bruit).
    d2 = np.diff(sma, n=2)
    bruit = (1.4826 * np.median(np.abs(d2 - np.median(d2)))) / np.sqrt(6) + 1e-12

    sw = np.lib.stride_tricks.sliding_window_view(sma, w)
    med = np.median(sw, axis=1)
    step = med[w:] - med[:-w]                       # mediane apres - mediane avant
    if len(step) < 10:
        return vide
    ech = 1.4826 * np.median(np.abs(step - np.median(step))) + 1e-12
    z = np.abs(step - np.median(step)) / ech

    idx = np.flatnonzero(z > seuil)
    if not len(idx):
        return vide
    ## un saut s'etale sur quelques TLE : on regroupe les indices contigus
    grp = [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > w // 2) + 1) if len(g)]
    ampl = np.array([np.abs(step[g]).max() for g in grp])
    return {"n_sauts": len(grp),
            "saut_median_bruit": float(np.median(ampl) / bruit),
            "amplitude_bruit": float(ampl.max() / bruit)}


@torch.no_grad()
def reconstruction_scores(mae, cfg, per_obj, feature_cols, norads, cols=("sma", "sma_diff"),
                          n_windows=6, batch_size=256, seed=0, device=None):
    """R2 de reconstruction sur patchs masques, par objet, pour les canaux `cols`.

    Passe par mae(x) en batch plutot que par reconstruct_window (une fenetre a la fois) :
    c'est le meme calcul de residu, mais des centaines de fois plus rapide, ce qui permet
    de classer des milliers d'objets pour en extraire les meilleurs.

    Renvoie un DataFrame {norad, R2_<col>..., R2_moyen, n_evenements}. n_evenements compte
    les pics de sma_diff au-dela de 8 ecarts-types robustes : un objet parfaitement plat se
    reconstruit trivialement bien, on veut pouvoir exiger qu'il se passe quelque chose.
    """
    device = device or next(mae.parameters()).device
    W = cfg.data.window_size
    F = len(feature_cols)
    idx_cols = [feature_cols.index(c) for c in cols]
    mae = mae.to(device).eval()

    fenetres, proprio, profils = [], [], {}
    i_sma = feature_cols.index("sma") if "sma" in feature_cols else None
    for n in norads:
        X = per_obj.get(n)
        if X is None or len(X) < W:
            continue
        if i_sma is not None:
            profils[n] = maneuver_profile(X[:, i_sma])
        for s0 in np.linspace(0, len(X) - W, n_windows).astype(int):
            fenetres.append(X[s0:s0 + W].T)
            proprio.append(n)

    if not fenetres:
        return pd.DataFrame(columns=["norad"])

    proprio = np.asarray(proprio)
    se = np.zeros((len(fenetres), len(idx_cols)))
    st = np.zeros((len(fenetres), len(idx_cols)))
    torch.manual_seed(seed)
    for i in range(0, len(fenetres), batch_size):
        xb = torch.from_numpy(np.stack(fenetres[i:i + batch_size])).float().to(device)
        pred, target = mae(xb)
        B, N, _ = pred.shape
        d = (pred - target).reshape(B, N, F, -1)[:, :, idx_cols, :]
        t = target.reshape(B, N, F, -1)[:, :, idx_cols, :]
        se[i:i + B] = d.pow(2).mean(dim=(1, 3)).double().cpu().numpy()
        st[i:i + B] = t.pow(2).mean(dim=(1, 3)).double().cpu().numpy()

    lignes = []
    for n in np.unique(proprio):
        m = proprio == n
        r2 = {}
        for j, c in enumerate(cols):
            num, den = se[m, j].mean(), st[m, j].mean()
            r2[f"R2_{c}"] = 1 - num / den if den > 1e-12 else np.nan
        lignes.append({"norad": int(n), **r2,
                       "R2_moyen": float(np.nanmean(list(r2.values()))),
                       **profils.get(n, {})})
    return pd.DataFrame(lignes).sort_values("R2_moyen", ascending=False).reset_index(drop=True)


def maneuver_indices(sma, w=10, seuil=10.0):
    """Indices des sauts persistants de sma (meme detection que maneuver_profile)."""
    sma = np.asarray(sma, dtype=np.float64)
    if len(sma) < 4 * w:
        return np.array([], dtype=int)
    sw = np.lib.stride_tricks.sliding_window_view(sma, w)
    med = np.median(sw, axis=1)
    step = med[w:] - med[:-w]
    if len(step) < 10:
        return np.array([], dtype=int)
    ech = 1.4826 * np.median(np.abs(step - np.median(step))) + 1e-12
    idx = np.flatnonzero(np.abs(step - np.median(step)) / ech > seuil)
    if not len(idx):
        return np.array([], dtype=int)
    grp = [g for g in np.split(idx, np.flatnonzero(np.diff(idx) > w // 2) + 1) if len(g)]
    ## step[i] compare [i, i+w) a [i-w, i) une fois decale de w : le saut est en i+w
    return np.array([int(g[np.abs(step[g]).argmax()]) + w for g in grp])


@torch.no_grad()
def maneuver_reconstruction_score(mae, cfg, per_obj, feature_cols, norad,
                                  col="sma", max_evenements=4, device=None):
    """Qualite de reconstruction AUX MANOEUVRES, et non moyennee sur toute la fenetre.

    Le R2 global recompense une ligne de base facile : la quasi-totalite des patchs est de
    la decroissance calme, et un objet au sma tres lisse obtient un excellent R2 sans que le
    modele ait jamais reproduit un saut. Ici on force le masquage du patch qui CONTIENT le
    saut et on mesure la reconstruction sur ce seul patch.

    R2 proche de 1 : le modele restitue la marche. Negatif : il l'a lissee en interpolant.
    """
    device = device or next(mae.parameters()).device
    W, P = cfg.data.window_size, cfg.model.patch_size
    N = W // P
    icol = feature_cols.index(col)
    X = per_obj.get(norad)
    if X is None or len(X) < W:
        return np.nan, 0

    evs = maneuver_indices(X[:, feature_cols.index("sma")])
    evs = [e for e in evs if 0 <= e < len(X)][:max_evenements]
    if not evs:
        return np.nan, 0

    scores = []
    for e in evs:
        ## fenetre centree sur l'evenement, recadree aux bords
        s0 = int(np.clip(e - W // 2, 0, len(X) - W))
        pos = e - s0
        p_ev = int(np.clip(pos // P, 0, N - 1))
        ## on masque le patch de l'evenement + de quoi atteindre le taux du pretraining
        n_mask = max(1, round(N * mae.masking_ratio))
        autres = [p for p in range(N) if p != p_ev]
        rng = np.random.default_rng(int(e))
        mp = np.sort(np.concatenate([[p_ev], rng.choice(autres, n_mask - 1, replace=False)]))

        x = X[s0:s0 + W].T
        rec, _ = reconstruct_window(mae, x, masked_patches=mp, device=device, seed=0)
        a, b = p_ev * P, (p_ev + 1) * P
        vrai = x.T[a:b, icol]
        pred = rec.T[a:b, icol]
        den = np.sum((vrai - vrai.mean()) ** 2)
        if den > 1e-12:
            scores.append(1 - np.sum((vrai - pred) ** 2) / den)
    return (float(np.mean(scores)), len(scores)) if scores else (np.nan, 0)


def export_reconstructions(mae, cfg, stats, per_obj, feature_cols, out_dir,
                           n_representatifs=100, n_meilleurs=10, n_windows=3,
                           cols=("sma", "sma_diff"), min_sauts=3, min_amplitude=200.0,
                           candidats=None, toujours_inclure=(), seed=0):
    """Ecrit les figures de reconstruction dans out_dir/ et out_dir/best_reconstructions/.

    - representatifs : tirage aleatoire uniforme sur les objets exploitables, pour donner
      une image non biaisee de ce que vaut le modele sur le catalogue ;
    - meilleurs : classes par R2 de reconstruction, mais uniquement parmi les objets dont le
      profil montre au moins `min_sauts` manoeuvres discretes d'amplitude `min_amplitude`
      fois le bruit (cf. maneuver_profile). Sans ce filtre le classement remonte les Starlink
      en montee d'orbite : une rampe continue de 400 km, trivialement lisse a reconstruire,
      qui obtient un excellent R2 sans contenir la moindre manoeuvre.
    """
    out_dir = Path(out_dir)
    best_dir = out_dir / "best_reconstructions"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    W = cfg.data.window_size
    exploitables = [n for n, X in per_obj.items() if len(X) >= W]
    rng = np.random.default_rng(seed)

    representatifs = list(rng.choice(exploitables, size=min(n_representatifs, len(exploitables)),
                                     replace=False))
    for n in toujours_inclure:
        if n in per_obj and n not in representatifs:
            representatifs.append(n)

    print(f"{len(exploitables)} objets exploitables -> {len(representatifs)} figures representatives")
    for i, n in enumerate(representatifs, 1):
        plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad=int(n),
                            n_windows=n_windows, to_plot=cols,
                            save_path=out_dir / f"{int(n)}.png")
        if i % 25 == 0:
            print(f"  {i}/{len(representatifs)}", flush=True)

    candidats = list(candidats) if candidats is not None else exploitables
    print(f"classement de {len(candidats)} candidats...", flush=True)
    sc = reconstruction_scores(mae, cfg, per_obj, feature_cols, candidats, cols=cols)
    retenus = sc[(sc.n_sauts >= min_sauts) & (sc.saut_median_bruit >= min_amplitude)].copy()
    print(f"  {len(retenus)}/{len(sc)} candidats passent le filtre "
          f"(>= {min_sauts} sauts, amplitude >= {min_amplitude:g}x le bruit)", flush=True)
    if len(retenus) < n_meilleurs:          # critere trop strict pour ce vivier
        retenus = sc[sc.n_sauts >= min_sauts].copy()

    ## classement final sur la reconstruction AUX manoeuvres, pas sur le R2 global
    print(f"  evaluation aux manoeuvres sur {len(retenus)} objets...", flush=True)
    vals = [maneuver_reconstruction_score(mae, cfg, per_obj, feature_cols, int(n))
            for n in retenus.norad]
    retenus["R2_manoeuvre"] = [v[0] for v in vals]
    retenus["n_evalues"] = [v[1] for v in vals]
    retenus = retenus.sort_values("R2_manoeuvre", ascending=False)
    sc = sc.merge(retenus[["norad", "R2_manoeuvre", "n_evalues"]], on="norad", how="left")
    meilleurs = retenus.head(n_meilleurs)

    for n in meilleurs.norad:
        plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad=int(n),
                            n_windows=n_windows, to_plot=cols,
                            save_path=best_dir / f"{int(n)}.png")
    for n in toujours_inclure:
        if n in per_obj:
            plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad=int(n),
                                n_windows=n_windows, to_plot=cols,
                                save_path=best_dir / f"{int(n)}.png")

    sc.to_csv(out_dir / "scores.csv", index=False)
    print(f"\n-> {out_dir} : {len(representatifs)} figures + scores.csv")
    print(f"-> {best_dir} : {len(meilleurs)} meilleures reconstructions")
    return sc, meilleurs


## Clusters de constellations

## Familles reconnues dans les noms SpaceTrack. L'ordre compte : le premier motif present
## dans le nom l'emporte, donc les motifs specifiques doivent preceder les generiques.
FAMILLES = ("STARLINK", "ONEWEB", "QIANFAN", "KUIPER", "GUOWANG", "FLOCK", "SKYSAT",
            "LEMUR", "SPIRE", "IRIDIUM", "GLOBALSTAR", "ORBCOMM", "YAOGAN", "JILIN",
            "GAOFEN", "CENTISPACE", "HAWK", "ICEYE", "PLANET", "DOVE", "NUSAT",
            "SENTINEL", "LANDSAT", "SPOT", "METOP", "NOAA", "COSMOS", "TBA")


def famille_du_nom(nom):
    """Famille deduite du nom SpaceTrack.

    Trois cas sont traites AVANT les constellations, sinon ils les polluent :
      - "... DEB" : fragment de debris. Le nom porte celui de l'objet parent, donc le
        detecter apres les familles ferait passer "COSMOS 1408 DEB" pour un satellite Cosmos ;
      - "... R/B"  : etage de lanceur, meme probleme ;
      - "TBA - TO BE ASSIGNED" : objet catalogue mais pas encore identifie. Ce n'est pas une
        constellation : le marquer comme tel evite des dossiers nommes "tba".
    """
    n = (nom or "").upper()
    if "TBA" in n or "TO BE ASSIGNED" in n:
        return "NON_IDENTIFIE"
    if " DEB" in n or n.endswith("DEB") or "DEBRIS" in n:
        return "DEBRIS"
    if "R/B" in n or "ROCKET BODY" in n:
        return "ETAGE"
    for f in FAMILLES:
        if f in n:
            return f
    mots = n.replace("-", " ").split()
    return mots[0] if mots else "INCONNU"


def nommer_cluster(familles, purete_min=30.0):
    """Nom d'un cluster a partir de la distribution de familles de ses membres.

    Les objets non identifies sont ecartes du vote : un cluster de Starlink recents dont la
    moitie n'a pas encore de nom reste un cluster Starlink. S'il ne reste rien d'identifie,
    ou si la famille dominante est trop minoritaire, le cluster est dit mixte plutot que de
    porter le nom d'une famille qui n'y est pas majoritaire.
    """
    v = familles.value_counts()
    identifiees = v.drop(labels=["NON_IDENTIFIE", "INCONNU"], errors="ignore")
    if identifiees.empty:
        return "non_identifie", 100.0 * v.get("NON_IDENTIFIE", 0) / max(len(familles), 1)
    purete = 100.0 * identifiees.iloc[0] / identifiees.sum()
    if purete < purete_min:
        deux = "_".join(f.lower() for f in identifiees.index[:2])
        return f"mixte_{deux}", purete
    return identifiees.index[0].lower(), purete


def object_names(norads, data_dir=None):
    """{norad: object_name}, lu directement dans les parquets (une seule ligne par objet)."""
    import pyarrow.dataset as pads
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "spacetrack" / "leo_payloads_and_debris"
    d = pads.dataset(str(data_dir), format="parquet")
    t = d.to_table(columns=["norad", "object_name"],
                   filter=pads.field("norad").isin([int(n) for n in norads]))
    df = t.to_pandas().drop_duplicates("norad")
    return dict(zip(df.norad.astype(int), df.object_name))


def backbone_from_mae(mae, cfg, device=None):
    """VanillaViT portant les poids de l'encodeur du MAE deja charge.

    Evite de relire le checkpoint sur disque comme le fait load_pretrained_backbone :
    on a deja le modele en memoire.
    """
    from ml.model import VanillaViT
    device = device or next(mae.parameters()).device
    bb = VanillaViT(
        n_features=mae.patch_embedding.n_features,
        n_epochs=mae.patch_embedding.window_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.encoder_embed_dim,
        n_attn_heads=cfg.model.encoder_n_attn_heads,
        n_blocks=cfg.model.encoder_n_blocks,
        expansion_factor=cfg.model.expansion_factor,
        dropout_rate=cfg.model.dropout_rate,
    )
    _, inattendus = bb.load_state_dict(mae.encoder_state_dict(), strict=False)
    if inattendus:
        raise RuntimeError(f"poids inattendus dans l'encodeur : {inattendus}")
    return bb.to(device).eval()


def export_clusters(mae, cfg, stats, per_obj, feature_cols, out_dir,
                    n_objets=6000, stride=256, min_cluster_size=25, min_samples=10,
                    n_figures=8, n_clusters_max=15, cols=("sma", "sma_diff"), seed=0):
    """Un sous-dossier de figures par cluster de constellation.

    Les representations viennent du token CLS de l'encodeur pre-entraine, moyenne sur les
    fenetres de l'objet, puis HDBSCAN (label -1 = bruit, non exporte). Chaque cluster est
    nomme d'apres la famille majoritaire de ses membres.

    Les figures retenues sont les objets les PLUS PROCHES DU CENTROIDE du cluster : ce sont
    les representants typiques, pas des membres de bordure qui donneraient une image faussee
    de ce que le cluster contient.
    """
    device = next(mae.parameters()).device
    W = cfg.data.window_size
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    eligibles = [n for n, X in per_obj.items() if len(X) >= W]
    choisis = list(rng.choice(eligibles, size=min(n_objets, len(eligibles)), replace=False))
    sous = {int(n): per_obj[n] for n in choisis}
    print(f"representations sur {len(sous)} objets (fenetre {W}, stride {stride})...", flush=True)

    bb = backbone_from_mae(mae, cfg, device)
    emb = embed_objects(bb, sous, device, window_size=W, stride=stride)
    print(f"  {len(emb)} representations -> HDBSCAN", flush=True)

    X_scaled, labels, norad_ids = cluster_embeddings(
        emb, min_cluster_size=min_cluster_size, min_samples=min_samples)
    norad_ids = np.asarray(norad_ids)
    labels = np.asarray(labels)

    noms = object_names(norad_ids)
    valides = sorted([c for c in set(labels) if c != -1],
                     key=lambda c: -(labels == c).sum())[:n_clusters_max]
    print(f"  {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
          f"{100*(labels == -1).mean():.1f}% de bruit -> export des {len(valides)} plus gros",
          flush=True)

    recap = []
    for rang, c in enumerate(valides, 1):
        m = labels == c
        membres = norad_ids[m]
        fam = pd.Series([famille_du_nom(noms.get(int(n), "")) for n in membres])
        nom_fam, purete = nommer_cluster(fam)
        dominante = fam.value_counts()

        centre = X_scaled[m].mean(axis=0)
        dist = np.linalg.norm(X_scaled[m] - centre, axis=1)
        proches = membres[np.argsort(dist)[:n_figures]]

        dossier = out_dir / f"{rang:02d}_{nom_fam}_n{len(membres)}"
        dossier.mkdir(parents=True, exist_ok=True)
        for n in proches:
            plot_object_windows(mae, cfg, stats, per_obj, feature_cols, norad=int(n),
                                n_windows=1, to_plot=cols,
                                save_path=dossier / f"{int(n)}_{famille_du_nom(noms.get(int(n),'')).lower()}.png")
        pd.DataFrame({"norad": membres,
                      "nom": [noms.get(int(n), "") for n in membres],
                      "famille": fam.values}).to_csv(dossier / "membres.csv", index=False)
        recap.append({"dossier": dossier.name, "taille": len(membres),
                      "famille": nom_fam, "purete_%": purete,
                      "composition": ", ".join(f"{k} {100*v/len(membres):.0f}%"
                                               for k, v in dominante.head(3).items())})
        print(f"  {dossier.name:34s} {len(membres):5d} obj | {dominante.head(3).to_dict()}", flush=True)

    rec = pd.DataFrame(recap)
    rec.to_csv(out_dir / "clusters.csv", index=False)
    print(f"\n-> {out_dir} : {len(valides)} sous-dossiers")
    return rec, pd.DataFrame({"norad": norad_ids, "cluster": labels,
                              "nom": [noms.get(int(n), "") for n in norad_ids]})


## ---------------------------------------------------------------------------------------
## SCRIPT : repartition de la loss par canal, avec la config qui l'a produite
##
##     python src/ml/mae_reconstruction.py --ckpt 2026-08-31_12-05-38
##
## La table seule est ininterpretable : un canal a 0 % peut l'etre parce qu'il est bien
## reconstruit, ou parce que son poids est nul et qu'on ne le regarde tout simplement pas.
## D'ou l'impression systematique de la config a cote des chiffres.

def print_run_config(ckpt_path, mae, cfg, feature_cols):
    """Imprime la config du checkpoint, puis ce qu'elle implique et qui ne s'y lit pas."""
    from omegaconf import OmegaConf

    meta = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cols_ckpt = meta.get('feature_cols')

    print("=" * 88)
    print(f"CHECKPOINT  {ckpt_path}")
    print("=" * 88)
    print(f"  epoch {meta.get('epoch')} | n_features {meta.get('n_features')} | "
          f"window_size {meta.get('window_size')} | scaler_kind {meta.get('scaler_kind', 'global')}")
    print(f"  feature_cols du checkpoint : "
          f"{list(cols_ckpt) if cols_ckpt is not None else 'ABSENT (checkpoint anterieur a leur sauvegarde)'}")
    print(f"  feature_cols recalculees   : {list(feature_cols)}")
    if cols_ckpt is not None and list(cols_ckpt) != list(feature_cols):
        print("  /!\\ ELLES DIFFERENT : les chiffres ci-dessous portent sur d'autres canaux "
              "que ceux du pretrain")

    print("\n" + "-" * 88)
    print("CONFIG COMPLETE DU RUN")
    print("-" * 88)
    print(OmegaConf.to_yaml(cfg).rstrip())

    print("\n" + "-" * 88)
    print("CE QUE LA CONFIG IMPLIQUE (et qui ne se lit pas directement)")
    print("-" * 88)
    W, P = cfg.data.window_size, cfg.model.patch_size
    N = W // P
    n_masques = round(N * cfg.model.masking_ratio)
    print(f"  patchs           : {W} / {P} = {N} patchs, masking {cfg.model.masking_ratio} "
          f"-> {n_masques} masques et {N - n_masques} visibles par fenetre")
    print(f"  la loss ne porte QUE sur les {n_masques} patchs masques")
    print(f"  normalisation dataset : {'par objet' if cfg.data.get('revin_norm') else 'globale'}")

    ## lu dans le MODELE et non dans la config : c'est le buffer qui agit reellement
    actifs = [c for c, m in zip(feature_cols, mae.inorm_mask.tolist()) if m]
    planchers = {c: round(f, 6) for c, f in zip(feature_cols, mae.sigma_floor.tolist()) if f}
    print(f"  RevIN par fenetre (buffers du modele) : {actifs or 'aucun canal'} "
          f"| planchers sigma {planchers or '-'}")

    print("\n  poids de loss effectifs (un canal absent de task.channel_weights garde 1.0) :")
    w = loss_weights(cfg, feature_cols)
    declares = set((cfg.task.get("channel_weights") or {}).items())
    declares_noms = {c for c, _ in declares}
    for canal, poids in zip(feature_cols, w):
        origine = "config" if canal in declares_noms else "defaut"
        print(f"      {canal:<12} {poids:>5.2f}   ({origine})")
    implicites = [c for c in feature_cols if c not in declares_noms]
    if implicites:
        print(f"  /!\\ {len(implicites)} canaux ne sont pas listes dans task.channel_weights "
              f"et pesent donc 1.0 sans qu'on l'ait choisi : {implicites}")
    inconnus = [c for c in declares_noms if c not in feature_cols]
    if inconnus:
        print(f"  /!\\ {inconnus} figurent dans channel_weights mais pas dans les features : ignores ici")


def main():
    import argparse
    from omegaconf import OmegaConf

    parseur = argparse.ArgumentParser(
        description="Repartition de la loss du MAE par canal, avec la config du run.")
    parseur.add_argument("--ckpt", default="2026-08-31_12-05-38",
                         help="identifiant du run sous outputs/ml/pretrain/")
    parseur.add_argument("--n-windows", type=int, default=2048,
                         help="fenetres tirees au hasard pour estimer les MSE par canal")
    parseur.add_argument("--batch-size", type=int, default=256)
    parseur.add_argument("--seed", type=int, default=0)
    parseur.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parseur.add_argument("--csv", action="store_true",
                         help="ecrit aussi channel_loss.csv dans le dossier du run")
    args = parseur.parse_args()

    base = Path(__file__).resolve().parents[2]
    run_dir = base / "outputs" / "ml" / "pretrain" / args.ckpt
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"introuvable : {ckpt_path}\n"
                         f"runs disponibles : {sorted(p.name for p in (base / 'outputs/ml/pretrain').iterdir())}")

    mae, cfg, stats, per_obj, feature_cols = load_mae_and_data(args.ckpt, base_path=base)
    print_run_config(ckpt_path, mae, cfg, feature_cols)

    print("\n" + "-" * 88)
    print(f"REPARTITION DE LA LOSS  ({args.n_windows} fenetres tirees dans {len(per_obj)} objets, "
          f"seed {args.seed}, device {args.device})")
    print("-" * 88)
    table = channel_loss_breakdown(mae, cfg, per_obj, feature_cols, n_windows=args.n_windows,
                                   batch_size=args.batch_size, seed=args.seed, device=args.device)
    with pd.option_context('display.float_format', lambda v: f"{v:10.6f}"):
        print(table.to_string(index=False))

    ## Lecture : ce que la table dit et qu'on lirait de travers sans le rappeler.
    print()
    cumul = table['part_%'].cumsum()
    k = int((cumul < 90).sum()) + 1
    tetes = ", ".join(table['canal'].head(k))
    print(f"  {k} canal(aux) font {cumul.iloc[k - 1]:.1f} % de la loss : {tetes}")

    nuls = table[table['poids'] == 0]
    if not nuls.empty:
        part_brute = nuls['mse'].sum() / table['mse'].sum() * 100
        print(f"  canaux a poids nul ({', '.join(nuls['canal'])}) : {part_brute:.1f} % de la MSE "
              f"BRUTE, 0 % de ce qui est optimise -- le modele ne les apprend pas, "
              f"la loss ne le dira jamais")
    print(f"  loss reconstituee : {table['contribution'].sum():.6f}  "
          f"(a comparer a la val loss du run dans metrics.jsonl)")

    if args.csv:
        sortie = run_dir / "channel_loss.csv"
        table.to_csv(sortie, index=False)
        print(f"\n  ecrit : {sortie}")


if __name__ == "__main__":
    main()
