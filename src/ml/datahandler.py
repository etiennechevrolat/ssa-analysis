
from pathlib import Path
import pandas as pd
import numpy as np
import os
from ssa.orbital import to_equinoxal, to_keplerian, mean_to_true_anomaly
from statsmodels.robust import scale

## Récupération des données sous la forme du dataset SPLID : 2000 .csv par satellite, avec pleins de params orbitaux.
def load_splid_objects(data_dir: Path, labels_path : Path | None = None):
    """
    Charge les csv splid en {object_id, df}
    """
    data_dir = Path(data_dir)
    
    labels = pd.read_csv(labels_path) if labels_path is not None else None
    objects = {}
    for csv in sorted(data_dir.glob("*.csv")): ## tri pour garantir reproductibilité, méthode path.glob(motif) cherche des motif dans le path 
        df = pd.read_csv(csv) # dataframe de l'objet considéré
        df["TimeIndex"]=range(len(df)) # création d'une colonne timeindex. unité dans laquelle les labels repèrent les manoeuvres.
        objects[int(csv.stem)] = df # .Stem est le nom du fichier sans le dossier ni l'extension
    if not objects:
        raise FileNotFoundError(f"aucun csv trouvé dans {data_dir}")
    return objects, labels 

## Récupération des données SPACETRACK pour inférence, une fois le modèle entrainé. Les fonctions suivantes le remette au format des données SPLID
### Les données spacetrack sont récupérées au format parquet avec les colonnes suivantes : 
## ['norad', 'object_name', 'epoch', 'creation_date', 'rev_at_epoch', 'inclination', 'raan', 'arg_perigee', 'mean_anomaly', 'mean_motion', 'eccentricity', 'bstar', 'sma', 'apogee', 'period', 'velocity']
# Il faut à la fois renommer les bonnes colonnes, et reprojetter une série irrégulière de TLE Spacetrack sur une grille régulière.

spacetrack_to_splid_rename = {
    "eccentricity" : "Eccentricity", 
    "inclination" : "Inclination (deg)",
    "raan" :  "RAAN (deg)",
    "arg_perigee" : "Argument of Periapsis (deg)",
    "mean_anomaly" : 'Mean Anomaly (deg)'
    }

SPLID_CADENCE_HOURS = 2.0 # 1 TimeIndex Splid = 2h 


def _regularize(df, cadence_hours, max_gap_hours):
    """
    Epochs tle irrégulières -> grille régulière 
    """
    df = df.sort_values(['epoch', 'creation_date']).drop_duplicates('epoch', keep='last')
    epochs = pd.to_datetime(df['epoch'])
    t= epochs.to_numpy('datetime64[ns]').astype('int64') / 1e9

    equinoxal= to_equinoxal(df['Eccentricity'], df['Inclination (deg)'], df['RAAN (deg)'], df['Argument of Periapsis (deg)'], df['Mean Anomaly (deg)'], anomaly_type = 'mean')

    freq = pd.Timedelta(hours=cadence_hours)
    grid = pd.date_range(epochs.iloc[0].ceil(freq), epochs.iloc[-1].floor(freq), freq=freq)
    tg = grid.to_numpy('datetime64[ns]').astype('int64') / 1e9

    k_g, h_g, q_g, p_g, cosM_g, sinM_g = (np.interp(tg, t, v) for v in equinoxal)
    a_g = np.interp(tg, t,  df['sma'].to_numpy(float) * 1000 ) # km ->m 

    ## pour l'interpolation de cos et sin, on renormalise
    norm = np.hypot(cosM_g, sinM_g)
    cosM_g, sinM_g = cosM_g / norm, sinM_g / norm

    e_g, i_g, raan_g, argp_g, M_g = to_keplerian(k_g, h_g, q_g, p_g, cosM_g, sinM_g)

    j = np.searchsorted(t,tg, side='right') - 1

    return pd.DataFrame({
        'TimeStamp' : grid,
        'Eccentricity' : e_g, 
        'Semimajor Axis (m)' : a_g,
        'Inclination (deg)' : i_g,
        'RAAN (deg)' : raan_g,
        'Argument of Periapsis (deg)' : argp_g,
        'Mean Anomaly (deg)' : M_g, 
        'gap' : (t[(j+1).clip(max=len(t) -1)] -t[j]) / 3600.0 > max_gap_hours, 
        'TimeIndex' : range(len(grid)),
    }
    )

def split_on_gaps(objects, min_length = 1):
    """
    Découpe en segments contigus sans trou > max_gap_hours
    """
    segments = {}
    for oid, df in objects.items():
        gap = df['gap'].to_numpy(bool)
        block = np.cumsum(gap) # les points 'gap' coupent la série
        for i, (_, seg) in enumerate(df[~gap].groupby(block[~gap])):
            if len(seg) < min_length : 
                continue
            seg= seg.reset_index(drop=True)
            seg['TimeIndex'] = range(len(seg))
            segments[f"{oid}_{i}"] = seg
    return segments


def load_spacetrack_objects_to_splid(data_dir : Path, out_dir = None, cadence_hours=SPLID_CADENCE_HOURS, max_gap_hours=24.0):

    data_dir = Path(data_dir)
    out_dir = Path(out_dir) 

    paths = sorted(data_dir.glob("*geo_validation.parquet"))

    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    raw = raw.rename(columns = spacetrack_to_splid_rename)
    objects={}
    
    out_dir.mkdir(parents=True, exist_ok=True)

    for norad, sub in raw.groupby('norad') : 
        
        csv_path =out_dir/ "geo_splid_format" /  f"{norad}.csv"
        df = _regularize(sub, cadence_hours, max_gap_hours=max_gap_hours)
        df.to_csv(csv_path, index=False)

        objects[int(norad)] = df
        
    return objects


max_dt_hours = 12.0 

def load_doris_objects(data_dir):
    """
    Charge les csv du dataset doris en {object_id, df}
    """

    labels_dir = Path(os.path.join(data_dir,'leo_maneuvers_label.csv'))
    train_dir = Path(os.path.join(data_dir, 'train', 'leo_doris_orbital_params.csv'))

    labels = pd.read_csv(labels_dir)
    df_labels = labels.copy()
    df_labels['epoch_sec'] = pd.to_datetime(df_labels['epoch'], format = 'ISO8601').to_numpy('datetime64[ns]').astype('int64') / 1e9 ## epochs maneuvres en secondes

    df = pd.read_csv(train_dir)
    objects= {}
    df_labels['TimeIndex'] = np.nan

    for norad, sub in df.groupby('norad'):
        df_object = sub.sort_values(['epoch', 'creation_date']).drop_duplicates('epoch', keep='last').copy() ## on retire les doublons.
        df_object["TimeIndex"]=range(len(df_object))

        epochs_tle = pd.to_datetime(df_object['epoch']).to_numpy('datetime64[ns]').astype('int64') / 1e9 ## epochs en secondes
        df_object['epoch_sec'] = epochs_tle
        df_object["dt"] = np.log1p(np.diff(epochs_tle, prepend=epochs_tle[0])).astype(np.float32) 
        objects[int(norad)] = df_object

        mask = df_labels['norad_id'] == norad
        if not mask.any() : 
            continue 
        labels_epochs = df_labels.loc[mask, 'epoch_sec'].to_numpy()

        ## On prend le premier TLE POSTERIEUR à la manoeuvre, et non le plus proche :
        ## la signature d'une manoeuvre est dans sma_diff[i] = sma[i] - sma[i-1], donc elle
        ## n'apparaît qu'au premier TLE qui suit. Centrer la cible sur un TLE antérieur
        ## la place sur un point où le saut n'a pas encore eu lieu.
        idx_tle = np.searchsorted(epochs_tle, labels_epochs, side='left')
        posterieur_existe = idx_tle < len(epochs_tle) ## sinon la manoeuvre est après le dernier TLE
        pick = np.clip(idx_tle, 0, len(epochs_tle) - 1)
        ok = posterieur_existe & (epochs_tle[pick] - labels_epochs <= max_dt_hours * 3600)
        df_labels.loc[mask, 'TimeIndex'] = np.where(ok, pick, np.nan)

        ## diagnostics : labels hors fenêtre TLE, et collisions (2 manoeuvres -> 1 seul TLE)
        n_dropped = int((~ok).sum())
        n_collisions = len(pick[ok]) - len(np.unique(pick[ok]))
        if n_dropped or n_collisions:
            print(f"[load_doris_objects] norad {norad}: {n_dropped}/{len(ok)} labels écartés "
                  f"(aucun TLE dans les {max_dt_hours}h suivant la manoeuvre), "
                  f"{n_collisions} collisions de TimeIndex")

    return objects, df_labels


### Ici on load les objets spacetrack pour pré entrainement non supervisé.

def load_spacetrack_objects(data_dir : Path, dataset = 'leo'):
    if dataset == 'leo':
        dataset_dir = Path(os.path.join(data_dir, "leo_unlabelled_dataset" ))
    elif dataset == 'leo_with_debris':
        dataset_dir = Path(os.path.join(data_dir, "leo_payloads_and_debris" ))
    elif dataset== 'starlink' : 
        dataset_dir = Path(os.path.join(data_dir, "starlink" ))
    else : ## tous les objets
        dataset_dir = Path(os.path.join(data_dir, "max_objects_all_regimes" ))

    parquet_paths = sorted(dataset_dir.glob('*.parquet'))
    print("Chargement des données...")
    raw = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    objects={}

    print("Début du traitement des données...")
    outliers_total = 0
    total_tles = 0
    bstar_factor = np.abs(np.median(raw['bstar']))

    for norad,sub in raw.groupby('norad'):
        ## traitement du dataframe spécifique aux données spacetrack : epochs non régulières
        df = sub.sort_values(['epoch', 'creation_date']).drop_duplicates('epoch', keep='last') ## on retire les doublons.

        ## retire les TLEs spacetrack outliers détectés sur le sma 
        times= pd.to_datetime(df['epoch']).to_numpy('datetime64[ns]').astype('int64') / 1e9 ## epochs en secondes
        is_outlier = np.array(sma_outliers_detection(norad, df, 0, len(times), n_from_mad=4)) ## nettoie une partie des outliers aberrants sur le demi-grand-axe
        ## is_outlier est un MASQUE booleen de la longueur de la serie : c'est sa somme qui
        ## compte les outliers, pas sa longueur. Et le denominateur doit etre le nombre de
        ## TLE AVANT suppression, sinon le taux est calcule sur une base amputee.
        outliers_total += int(is_outlier.sum())
        total_tles += len(is_outlier)
        df = df[~is_outlier]

        # Ajout de la feature temporelle avec passage en log
        times= pd.to_datetime(df['epoch']).to_numpy('datetime64[ns]').astype('int64') / 1e9 ## epochs en secondes
        df["dt"] = np.log1p(np.diff(times, prepend=times[0])).astype(np.float32) # distribution à queue lourde. on ajoute une feature temporelle

        ## transformation de bstar 
        df['bstar'] = np.asinh(df['bstar'] / bstar_factor)

        ## sauvegarde dans objects
        objects[int(norad)] = df
    print(f"Fin du traitement des données, {len(objects)} objets, "
          f"{outliers_total}/{total_tles} TLE retirés ({100 * outliers_total / max(total_tles, 1):.3f}% outliers)")
    return objects

def sma_outliers_detection(norad, df, start, end, n_from_mad = 3):
    df_norad = df[df['norad'] == norad].iloc[start:end]
    sma= df_norad['sma'].values
    median = np.median(sma)
    median_abs_deviation = scale.mad(sma)

    is_potential_outlier = np.abs(sma - median) > n_from_mad*median_abs_deviation 
    is_outlier = [False for _ in range(len(is_potential_outlier))]

    for t in range(2, len(is_potential_outlier) -2) :
        if is_potential_outlier[t] :
            if ((is_potential_outlier[t + 1] & is_potential_outlier[t+2]) 
                or (is_potential_outlier[t - 1] & is_potential_outlier[t-2])
                or (is_potential_outlier[t-1] & is_potential_outlier[t+1])
                ): 
                continue   ## le point n'est pas isolé : il fait partie d'une série d'au moins 3 TLE = signifiant 
            else : 
                is_outlier[t] = True
    return is_outlier

## Features engineering

log_cols = ['sma']
diff_cols_splid  =['Semimajor Axis (m)', 'q', 'p']
diff_cols_spacetrack = ['sma'] ## attention : le sma spacetrack est en km
heavy_tail_cols_spacetrack = ['sma_diff'] ## residus a queue lourde : cf add_robust_asinh

def add_continuous_angles(df : pd.DataFrame, spacetrack=True):
    """transforme les paramètres angulaires discontinus : 
    (e, i, RAAN, arg_perigee, M) en paramètres equinoxaux continus :
    (k, h , q , p, cos(lamda), sin(lamda))
    et les ajoutent au dataframe
    """
    if spacetrack : 
        e = df['eccentricity']
        i = df['inclination'] 
        RAAN = df['raan']
        arg_perigee = df['arg_perigee']
        anomaly = df['mean_anomaly']
        anomaly_type = 'mean'

    else: ## SPLID data nomenclature
        if 'True Anomaly (deg)' in df.columns:
            anomaly, anomaly_type = df['True Anomaly (deg)'], 'true'
        else: 
            anomaly, anomaly_type = df['Mean Anomaly (deg)'], 'mean'
        e = df['Eccentricity']
        i = df['Inclination (deg)'] 
        RAAN = df['RAAN (deg)']
        arg_perigee = df['Argument of Periapsis (deg)']
    
    k,h,q,p, cosM, sinM = to_equinoxal(e,i,RAAN, arg_perigee, anomaly, anomaly_type)
    df['k'] = k
    df['h'] = h
    df['q'] = q
    df['p'] = p
    df['cosM'] = cosM
    df['sinM'] = sinM
    return df

def add_diff(df, diff_cols = diff_cols_spacetrack):
    """ajoute les derivées discrètes = résidus des colonnes de diff_cols au dataframe.
    suppose d'avoir déjà transformée les coordonnées angulaires kepleriennes en coordonnées equinoxales pour éviter les sauts brutaux liés au passage 360° -> 1°.
    """
    df = df.copy()
    for col in diff_cols:
        v = df[col].to_numpy(dtype=np.float64)
        df[f"{col}_diff"]= np.diff(v, prepend=v[0])
    return df

def add_robust_asinh(df, cols, k=5.0):
    """Normalise des colonnes a queue lourde : echelle robuste par objet, puis compression.
    """
    df = df.copy()
    for col in cols:
        v = df[col].to_numpy(dtype=np.float64)
        s = float(scale.mad(v))
        if not np.isfinite(s) or s <= 1e-12:
            ## objet trop court, ou residus tous identiques : le MAD degenere a 0 et la
            ## division exploserait. On retombe sur l'ecart-type, puis sur 1.0.
            s = float(v.std())
            if not np.isfinite(s) or s <= 1e-12:
                s = 1.0
        ## k elargit la zone lineaire d'asinh : les pics jusqu'a ~k sigma gardent leur
        ## amplitude, seule la queue au-dela est comprimee. k=1 comprime tot, k grand
        ## tend vers le brut (et vers des gradients domines par quelques echantillons).
        df[col] = np.arcsinh(v / (k * s))
    return df

def add_log(df, log_cols):
    df = df.copy()
    for col in log_cols : 
        v = df[col].to_numpy(dtype=np.float64)
        df[f"log({col})"] = np.log(v)
    return df 

level_cols = ['sma']

def add_level_and_local_variations(df, level_cols):
    df = df.copy()
    for col in level_cols:
        v = df[col].to_numpy(dtype=np.float64)
        df[f"{col}_level"] = v.mean()
        df[f"{col}_local"] = v - v.mean()
    return df 

def build_features(df, spacetrack=True, log_features=False):
    if not spacetrack:
        return _build_features_splid(df)
    if log_features:
        return _build_features_spacetrack_log(df)

    df = df.copy()
    df = add_continuous_angles(df, spacetrack=True)
    df = add_diff(df, diff_cols_spacetrack)
    df = add_robust_asinh(df, heavy_tail_cols_spacetrack, k=5.0)

    feature_cols = (['dt', 'bstar', 'sma', 'k', 'h', 'p', 'q', 'cosM', 'sinM']
                    + [f"{c}_diff" for c in diff_cols_spacetrack])
    df[feature_cols] = df[feature_cols].astype(np.float32)
    return df, feature_cols


def _build_features_splid(df):
    """Variante SPLID (localizer / classifier) : grille reguliere 2h, sma en metres.

    Pas de feature dt, la cadence etant reguliere, et nomenclature de colonnes differente
    (cf. add_continuous_angles). Conservee pour dataset.py et inference.py.
    """
    df = df.copy()
    df = add_continuous_angles(df, spacetrack=False)
    df = add_diff(df, diff_cols_splid)

    feature_cols = ['k', 'h', 'p', 'q', 'cosM', 'sinM'] + [f"{c}_diff" for c in diff_cols_splid]
    df[feature_cols] = df[feature_cols].astype(np.float32)
    return df, feature_cols


def _build_features_spacetrack_log(df):
    """Variante spacetrack avec log(sma) et sans bstar. Conservee pour comparaison.

    Non utilisee par le pretrain courant : build_features(..., log_features=True).
    """
    df = df.copy()
    df = add_continuous_angles(df, spacetrack=True)
    df = add_log(df, log_cols)
    df = add_diff(df, diff_cols_spacetrack)
    df = add_robust_asinh(df, heavy_tail_cols_spacetrack)

    feature_cols = (['dt', 'sma', 'k', 'h', 'p', 'q', 'cosM', 'sinM']
                    + [f"{c}_diff" for c in diff_cols_spacetrack]
                    + [f"log({col})" for col in log_cols])
    df[feature_cols] = df[feature_cols].astype(np.float32)
    return df, feature_cols
