"""Différentes fonctions de test des méthodes statistiques de détection de manœuvres.
Jusqu'à maintenant, 3 sont implémentées :
    - MLF Keplerien (résidu de longitude moyenne, LOWESS)
    - MLF SGP4      (résidu de longitude inertielle propagée, LOWESS)
    - Kalman        (dérivée estimée du demi-grand axe + seuil χ² sur la NIS)
"""

import matplotlib.pyplot as plt

from maneuver_detection.mean_longitude_filter import MLF_LOWESS_KEPLERIAN, MLF_LOWESS_SGP4
from maneuver_detection.discrete_kalman_filter import detect_kalman

import os 
import pandas as pd
import csv
from pathlib import Path


def compare_methods(df, norad, threshold_sigma=4.0, min_distance_days=7.0,
                    var_Q=0.13, r=1.0, p0=1000.0, alpha=0.997):

    """Compare les méthodes MLF_KEPLERIAN, MLF_SGP4, LKF sur le même satellite, axe temps (epoch) commun.

    ``df`` doit être trié par epoch et contenir les colonnes nécessaires aux
    trois méthodes : ``epoch``, ``sma``, ``mean_motion``, ``raan``,
    ``arg_perigee``, ``mean_anomaly``, ``inclination``, ``eccentricity``.

    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    sig_kep, peaks_kep = MLF_LOWESS_KEPLERIAN(
        df, threshold_sigma=threshold_sigma,
        min_distance_days=min_distance_days, ax=axes[0])
    axes[0].set_title("MLF Keplerien (LOWESS)")

    sig_sgp, peaks_sgp = MLF_LOWESS_SGP4(
        df, norad, threshold_sigma=threshold_sigma,
        min_distance_days=min_distance_days, ax=axes[1])
    axes[1].set_title("MLF SGP4 (LOWESS)")

    epoch, sma_dot, nis, man_kal, thr = detect_kalman(
        df, var_Q=var_Q, r=r, p0=p0, alpha=alpha, ax=axes[2])

    axes[2].set_xlabel("epoch")
    fig.suptitle(f"Détection de manœuvres — NORAD {norad}")
    plt.tight_layout()
    plt.show()
    
    return fig

import re
from optimisation_seuil.metrics import confusion_matrix, precision_recall_f1
import numpy as np
import polars as pl 

from optimisation_seuil.optimise_seuil import optimise_seuil_kalman_via_regul_gaussienne

def to_days(times, t0):
    """datetimes (np.datetime64 / datetime) -> float jours depuis t0."""
    arr = np.asarray(times, dtype="datetime64[ns]")
    return (arr - np.datetime64(t0, "ns")).astype("float64") / 8.64e13

def test_kalman_on_LEO():
    """
    Teste la méthode par filtation kalman sur des satellites geo d'intéret, labellisés à la main par mes soins.
    """
    ## Les paths des différentes sources de données
    base = os.path.dirname(os.path.abspath(__file__)) ## pointe vers folder parent, ici maneuver_detection

    true_man_path = Path(os.path.join(base, '..', '..', 'data', 'parsed', 'manual_labels_LEO'))

    path_tle_spacetrack = os.path.join(base, '..', '..', 'data', 'parsed', 'SPACETRACK', 'dataframe_rivesalt_from_spacetrack.parquet' )

    ### On récupère d'abord au format iso toutes les manoeuvres 
    true_maneuvers = {}

    for csv in sorted(true_man_path.glob("*.csv")):
        df = pd.read_csv(csv) ## de type pd.DataFrame
        m = re.search(r"output_ephemeride_(\d+)",csv.name )
        norad = int(m.group(1))
        
        true_maneuvers[norad] = df['date'] ## type Series de Panda

    ## On recupère les ids des satellites dont on a trouvé de la donnée spacetrack

    present = set(
        pl.scan_parquet(path_tle_spacetrack)
        .select("norad")
        .unique()
        .collect()
        .to_series().to_list()
        )

    for norad in true_maneuvers.keys():
        if norad not in present: 
            print(f"norad {norad} absent des données récups sur spacetrack")
            continue
        if true_maneuvers[norad].empty:
            print(f"norad {norad} ne manoeuvre pas")
            continue
        df = (
            pl.scan_parquet(path_tle_spacetrack)
            .filter(pl.col("norad") == norad)
            .select("epoch", "sma", "inclination")
            .sort("epoch")
            .collect()
        )

        ## Méthode kalman pour détection de manoeuvres in plane, on optimise satellite par satellite au préalable, pour avoir la meilleur f1 possible.
        alpha = 0.997
        metrics = optimise_seuil_kalman_via_regul_gaussienne(
            path_tle_spacetrack,
            true_maneuvers[norad],
            norad,
            man_ilrs = False,
            r_0 = 0.5, ## Params de départ pour l'optimisation
            varQ_0 = 0.05, ##
            p0_0 = 1000, ##
            sigma_lissage=0.9, ## Params de lissage
            beta_lissage=1.5,
            tol_days=3.0,
            metrique='sma'
            )

        epoch, sma_dot, nis, man, treshold = detect_kalman(
            df, 
            var_Q=metrics['var_Q'], 
            r=metrics['r'], 
            p0=metrics['p0'], 
            alpha=alpha, 
            plot=False)
        print(metrics)




    


test_kalman_on_LEO()