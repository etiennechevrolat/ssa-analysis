"""Comparaison des méthodes de détection de manœuvres sur un même satellite.

Empile les trois méthodes sur un axe temporel (epoch) commun et partagé, pour
juger si elles détectent les mêmes manœuvres aux mêmes dates :

    - MLF Keplerien (résidu de longitude moyenne, LOWESS)
    - MLF SGP4      (résidu de longitude inertielle propagée, LOWESS)
    - Kalman        (dérivée estimée du demi-grand axe + seuil χ² sur la NIS)
"""

import matplotlib.pyplot as plt

from .mean_longitude_filter import MLF_LOWESS_KEPLERIAN, MLF_LOWESS_SGP4
from .discrete_kalman_filter import detect_kalman


def compare_methods(df, norad, threshold_sigma=4.0, min_distance_days=7.0,
                    var_Q=0.13, r=1.0, p0=1000.0, alpha=0.997):
    """Trace les 3 méthodes sur le même satellite, axe temps (epoch) commun.

    ``df`` doit être trié par epoch et contenir les colonnes nécessaires aux
    trois méthodes : ``epoch``, ``sma``, ``mean_motion``, ``raan``,
    ``arg_perigee``, ``mean_anomaly``, ``inclination``, ``eccentricity``.

    Renvoie un dict des détections (indices + epochs) par méthode.
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

    return {
        "epoch": epoch,
        "keplerian": {"signal": sig_kep, "maneuvers": peaks_kep},
        "sgp4":      {"signal": sig_sgp, "maneuvers": peaks_sgp},
        "kalman":    {"sma_dot": sma_dot, "nis": nis, "maneuvers": man_kal},
    }
