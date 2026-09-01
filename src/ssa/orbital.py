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
        np.sqrt(1 - e**2) * np.sin(np.radians(nu)), 
        e + np.cos(np.radians(nu))
    )
    mean_anomaly = excentric_anomaly - e * np.sin(excentric_anomaly)
    return mean_anomaly

def mean_to_true_anomaly(mean_anomaly, e, n_iter=8) : 
    """
    M(deg) -> nu (deg)
    Equation de Kepler : M = E - e * sin(E). On retrouve E à partir de M avec l'algorithme de Newton
    """
    M = np.radians(np.asarray(mean_anomaly, dtype = float))
    e = np.asarray(e, dtype=float)
    E = M.copy()
    for _ in range(n_iter): 
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e)* np.cos(E/2))

    return np.degrees(nu) % 360.0

def _angles_radians(e, i, raan, arg_perigee, anomaly, anomaly_type):
    """Facteur commun aux deux conversions : degres -> radians et anomalie -> moyenne."""
    e = np.asarray(e, dtype=float)
    i = np.radians(np.asarray(i, dtype=float))
    raan = np.radians(np.asarray(raan, dtype=float))
    arg_perigee = np.radians(np.asarray(arg_perigee, dtype=float))

    if anomaly_type == 'true':
        M = true_to_mean_anomaly(anomaly, e)
    elif anomaly_type == 'mean':
        M = np.radians(np.asarray(anomaly, dtype=float))
    else:
        raise ValueError("Anomalie doit valoir 'true' ou 'mean'")
    return e, i, raan, arg_perigee, M


def to_equinoxal(e, i, raan, arg_perigee, anomaly, anomaly_type = 'true'):
    # Keplerian angles (e, i, RAAN, arg_perigee, M) -> Equinoxal (k, h , q , p, cos(M), sin(M)) continuous
    ## Conservee pour le repli sur les anciens checkpoints (legacy_angles) et pour le chemin
    ## SPLID. Le chemin principal passe desormais par to_separated_angles : ces coordonnees
    ## MELANGENT les elements (k et h portent e ET la longitude du pericentre, q et p portent
    ## l'inclinaison ET le RAAN), ce qui empeche d'attribuer une variation a un element.
    e, i, raan, arg_perigee, M = _angles_radians(e, i, raan, arg_perigee, anomaly, anomaly_type)

    longitude_pericentre = raan + arg_perigee
    k = e*np.cos(longitude_pericentre)
    h = e*np.sin(longitude_pericentre)
    q = np.tan(i/2)*np.cos(raan)
    p = np.tan(i/2)*np.sin(raan)
    return k,h,q,p, np.cos(M), np.sin(M)


def to_separated_angles(e, i, raan, arg_perigee, anomaly, anomaly_type='true'):
    """Elements kepleriens -> representation continue et SEPAREE, sans perte.

    Un element par grandeur physique, chaque angle rendu continu par son couple (cos, sin) :

        e                          excentricite, deja continue
        tan(i/2)                   inclinaison, monotone et sans repli sur (0, 180 deg)
        cos(RAAN), sin(RAAN)       noeud ascendant
        cos(argp), sin(argp)       argument du perigee
        cos(M), sin(M)             anomalie moyenne

    Difference avec to_equinoxal : les coordonnees equinoxiales compriment (e, argp, RAAN, i)
    en quatre nombres (k, h, q, p) ou chaque grandeur apparait dans plusieurs coordonnees.
    Une manoeuvre hors plan y bouge p ET q sans qu'on puisse dire lequel de i ou du RAAN a
    change. Ici chaque element a son canal, et rien n'est perdu : k = e cos(RAAN+argp),
    h = e sin(RAAN+argp), q = tan(i/2) cos(RAAN), p = tan(i/2) sin(RAAN) restent calculables.
    """
    e, i, raan, arg_perigee, M = _angles_radians(e, i, raan, arg_perigee, anomaly, anomaly_type)
    return (e, np.tan(i/2),
            np.cos(raan), np.sin(raan),
            np.cos(arg_perigee), np.sin(arg_perigee),
            np.cos(M), np.sin(M))

def to_keplerian(k,h,q,p, cos_M, sin_M): 
    #Equinoxal (k, h , q , p, cos(M), sin(M)) continuous ->  Keplerian angles (e, i, RAAN, arg_perigee, M) 
    e = np.hypot(h,k)
    i = 2 * np.arctan(np.hypot(q,p))
    varpi = np.arctan2(h,k)
    raan = np.arctan2(p,q)
    arg_perigee = varpi - raan 
    M = np.arctan2(sin_M, cos_M)

    return e, np.degrees(i), np.degrees(raan)%360.0, np.degrees(arg_perigee)%360.0, np.degrees(M)%360.0

