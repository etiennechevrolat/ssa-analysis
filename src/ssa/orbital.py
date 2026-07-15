import numpy as np 

# -- constantes physiques 
GM    = 398600441800000.0
GM13  = GM ** (1.0 / 3.0)
MRAD  = 6378.137
PI    = 3.14159265358979
TPI86 = 2.0 * PI / 86400.0

def derive(mean_motion, ecc):
    sma  = GM13 / ((TPI86 * mean_motion) ** (2/3)) / 1000.0
    smak = sma * 1000.0
    return {
        "sma": sma,
        "apogee":  sma * (1 + ecc) - MRAD,
        "period":  2 * PI * (smak**3 / GM) ** 0.5,
        "velocity": (GM / smak) ** 0.5,
    }

def true_to_mean_anomaly(nu, e):
    excentric_anomaly = np.arctan2(
        np.sqrt(1._- e**2) * np.sin(np.radians(nu)), 
        e + np.cos(np.radians(nu))
    )
    mean_anomaly = excentric_anomaly - e * np.sin(excentric_anomaly)
    return mean_anomaly
    
def to_equinoxal(e, i, raan, arg_perigee, true_anomaly):
    # Keplerian angles (e, i, RAAN, arg_perigee, M) from splid dataset -> Equinoxal (k, h , q , p, cos(lamda), sin(lamda)) continuous 

    e = np.asarray(e, dtype=float)
    i = np.radians(np.asarray(i, dtype=float))
    raan = np.radians(np.asarray(raan, dtype=float))
    arg_perigee = np.radians(np.asarray(arg_perigee, dtype=float))
    M = np.radians(np.asarray(true_to_mean_anomaly(true_anomaly)), dtype=float)
    

    longitude_pericentre = raan + arg_perigee
    k = e*np.cos(longitude_pericentre)
    h = e*np.sin(longitude_pericentre)
    q = np.tan(i/2)*np.cos(raan)
    p = np.tan(i/2)*np.sin(raan)
    return k,h,q,p, np.cos(M), np.sin(M)

def to_keplerian(k,h,q,p, cos_lamda, sin_lamda): 
    #Equinoxal (k, h , q , p, cos(lamda), sin(lamda)) continuous ->  Keplerian angles (e, i, RAAN, arg_perigee, M) 
    e = np.hypot(h,k)
    i = 2 * np.arctan(np.hypot(q,p))

    varpi = np.arctan2(h,k)
    raan = np.arctan2(p,q)

    arg_perigee = varpi - raan 
    M = np.arctan2(sin_lamda, cos_lamda) - varpi 

    return e, np.degrees(i), np.degrees(raan)%360.0, np.degrees(arg_perigee)%360.0, np.degrees(M)%360.0

"""
# Test
# racine du repo = 2 niveaux au-dessus de src/scripts/plot_history.py
ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "data" / "raw" / "STARLINK_US_1782475759.269574.parquet"

df = load_history(path, 
                  norad=48139, 
                  params=["semimajor_axis","eccentricity","inclination","raan","arg_perigee","mean_anomaly"]
)
print(to_equinoxal(df))
"""