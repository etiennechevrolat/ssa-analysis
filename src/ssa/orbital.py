
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
        "perigee": sma * (1 - ecc) - MRAD,
        "period":  2 * PI * (smak**3 / GM) ** 0.5,
        "velocity": (GM / smak) ** 0.5,
    }

