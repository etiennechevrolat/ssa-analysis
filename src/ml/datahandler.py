
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl 

from ssa.orbital import to_equinoxal, to_keplerian, mean_to_true_anomaly
 
diff_cols  =['Semimajor Axis (m)', 'q', 'p']



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


## Récupération des données SPACETRACK pour inférence, une fois le modèle entrainé. 
### Les données spacetrack sont récupérées au format parquet avec les colonnes suivantes : 
## ['norad', 'object_name', 'epoch', 'creation_date', 'rev_at_epoch', 'inclination', 'raan', 'arg_perigee', 'mean_anomaly', 'mean_motion', 'eccentricity', 'bstar', 'sma', 'apogee', 'period', 'velocity']
# Il faut à la fois renommer les bonnes colonnes, et reprojetter une série irrégulière de TLE Spacetrack sur une grille régulière.

spacetrack_to_splid_rename = {
    "eccentricity" : "Eccentricity", 
    "inclination" : "Inclination (deg)",
    "raan" :  "RAAN (deg)",
    "arg_perigee" : "Argument of Periapsis (deg)",
    }

SPLID_CADENCE_HOURS = 2.0 # 1 TimeIndex Splid = 2h 


def _regularize(df, cadence_hours, max_gap_hours):
    """
    Epochs tle irrégulières -> grille régulière 
    """
    df = df.sort_values(['epoch', 'creation_date']).drop_duplicates('epoch', keep='last')
    epochs = pd.to_datetime(df['epoch'])
    t= epochs.to_numpy('datetime64[ns]').astype('int64') / 1e9

    equinoxal= to_equinoxal(df['Eccentricity'], df['Inclination (deg)'], df['RAAN (deg)'], df['Argument of Periapsis (deg)'], df['mean_anomaly'], anomaly_type = 'mean')

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



def load_spacetrack_objects(data_dir : Path, cadence_hours=SPLID_CADENCE_HOURS, max_gap_hours=24.0):

    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*geo_validation.parquet"))

    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    raw = raw.rename(columns = spacetrack_to_splid_rename)
    objects={}
    for norad, sub in raw.groupby('norad') : 
        df = _regularize(sub, cadence_hours, max_gap_hours=max_gap_hours)
        objects[int(norad)] = df
    return objects




def add_continuous_angles(df : pd.DataFrame) -> pd.DataFrame:
    """transforme les paramètres angulaires discontinus : 
    (e, i, RAAN, arg_perigee, M) en paramètres equinoxaux continus :
    (k, h , q , p, cos(lamda), sin(lamda))
    et les ajoutent au dataframe
    df : pd.DataFrame contenant les paramètres d'un objet du dataset
    """
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

def add_diff(df, diff_cols = diff_cols):
    """ajoute les derivées discrètes = résidus des colonnes de diff_cols au dataframe.
    suppose d'avoir déjà transformée les coordonnées angulaires kepleriennes en coordonnées equinoxales pour éviter les sauts brutaux liés au passage 360° -> 1°.
    """
    df = df.copy()
    for col in diff_cols:
        v = df[col].to_numpy(dtype=np.float32)
        df[f"{col}_diff"]= np.diff(v, prepend=v[0])
    return df

def build_features(df, diff_cols = diff_cols):
    """
    Applique les transformations et renvois le dataframe enrichi des features précédentes utilisées par le modèle
    """
    df = df.copy()
    df = add_continuous_angles(df)
    df = add_diff(df, diff_cols)
    feature_cols = ['k','h', 'p', 'q', 'cosM', 'sinM' ] + [f"{c}_diff" for c in diff_cols]
    df[feature_cols] = df[feature_cols].astype(np.float32)

    return df, feature_cols

import os 
from pathlib import Path

def main() : 

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir_st = os.path.join(base, '..', '..', 'data', 'raw', 'spacetrack')
  
    objects = load_spacetrack_objects(data_dir_st)
    

if __name__ == "__main__": 
    main()