
from pathlib import Path
import pandas as pd
import numpy as np

diff_cols  =['Semimajor Axis (m)', 'Inclination (deg)', 'Latitude (deg)', 'Longitude (deg)', 'Altitude (m)']

def load_splid_objects(data_dir: Path, labels_path : Path | None = None):
    """
    charge les csv splid en {object_id, df}
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
from ssa.orbital import to_equinoxal, to_keplerian

def add_continuous_angles(df : pd.DataFrame) -> pd.DataFrame:
    """transforme les paramètres angulaires discontinus : 
    (e, i, RAAN, arg_perigee, M) en paramètres equinoxaux continus :
    (k, h , q , p, cos(lamda), sin(lamda))
    et les ajoutent au dataframe
    df : pd.DataFrame contenant les paramètres d'un objet du dataset
    """
    e = df['Eccentricity']
    i = df['Inclination (deg)'] 
    RAAN = df['RAAN (deg)']
    arg_perigee = df['Argument of Periapsis (deg)']
    nu = df['True Anomaly (deg)']
    
    k,h,q,p, cosM, sinM = to_equinoxal(e,i,RAAN, arg_perigee, nu)
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
