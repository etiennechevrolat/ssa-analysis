
import numpy as np
import polars as pl

from maneuver_detection.discrete_kalman_filter import kalman_filter_ordre_1

from scipy.stats import chi2

from optimisation_seuil.metrics import (
    lissage_noyau_gaussien_metriques,
    confusion_matrix,
    precision_recall_f1,
    minimize,
)


def to_days(times, t0):
    """datetimes (np.datetime64 / datetime) -> float jours depuis t0."""
    arr = np.asarray(times, dtype="datetime64[ns]")
    return (arr - np.datetime64(t0, "ns")).astype("float64") / 8.64e13

def optimise_seuil_kalman_via_regul_gaussienne(
        path_tle,
        path_man,
        norad,
        r_0 = 0.5, ## Params de départ pour l'optimisation
        varQ_0 = 0.05, ##
        p0_0 = 1000, ##
        sigma_lissage=1.0, ## Params de lissage
        beta_lissage=0.5,
        tol_days=1.0,
        metrique='sma'):

    # 1. Données TLE du satellite (epoch trié + métrique filtrée : sma ou inclination)
    alpha = 0.997
    df = (
        pl.scan_parquet(path_tle)
        .filter(pl.col("norad") == norad)
        .select("epoch", metrique)
        .sort("epoch")
        .collect()
    )

    if df.is_empty():
        raise ValueError(f"Aucune donnée TLE pour norad={norad}")

    # 2. Vérité terrain, restreinte à la fenêtre couverte par les données
    man = pl.scan_parquet(path_man).collect()
    true_dt = man["burn_epoch"].to_numpy()
    e = df["epoch"].to_numpy()
    true_dt = true_dt[(true_dt >= e[0]) & (true_dt <= e[-1])]

    # 3. Grille temporelle des candidats (fixe : ne dépend pas des paramètres)
    epoch = e[1:]                        # nis démarre au pas k = 1
    t0 = epoch[0]
    epoch_d = to_days(epoch, t0)         # temps candidats t_j
    true_d = to_days(true_dt, t0)        # manœuvres vraies t_i

    # 4. Loss lissée = fonction de x = [var_Q, r, p0].
    #    On relance le filtre à chaque évaluation : var_Q, r, p0 changent nis,

    def run_filter(var_Q, r, p0):
        _, nis, _ = kalman_filter_ordre_1(df, var_Q=var_Q, r=r, p0=p0, metrique=metrique)
        return np.asarray(nis, dtype=float)

    def loss(x):
        log_var_Q , log_r, log_p0 = x

        try:
            nis = run_filter(10**log_var_Q, 10**log_r, 10**log_p0)
        except Exception:
            return 1.0                       # filtre KO -> pire loss (F1 = 0)
        if not np.all(np.isfinite(nis)):
            return 1.0
        q = chi2.ppf(alpha, df=1)
        return lissage_noyau_gaussien_metriques(
            true_d, epoch_d, nis, q, sigma_lissage, beta_lissage
        )

    # 5. Optimisation des paramètres du filtre kalman : R , Variance de Q , P0
    x0 = np.array([np.log10(varQ_0), np.log10(r_0), np.log10(p0_0)])

    bounds = [(np.log10(1e-10), np.log10(10.0)),      # var_Q > 0
              (np.log10(1e-10), np.log10(100.0)),     # r  > 0
              (np.log10(1.0), np.log10(1e5)),        # p0 > 0
            ] 
    
    res = minimize(loss, x0=x0, bounds=bounds)
    log_var_Q, log_r, log_p0 = res.x
    var_Q, r, p0 = 10**log_var_Q, 10**log_r, 10**log_p0   # retour en espace linéaire

    # 6. Score aux paramètres optimaux : seuillage nis > q  -> F1
    nis_opt = run_filter(var_Q, r, p0)
    q_opt = chi2.ppf(alpha, df=1)

    pred_d = epoch_d[nis_opt > q_opt]
    conf = confusion_matrix(true_d, pred_d, tol_days=tol_days)
    prf = precision_recall_f1(conf)

    print(f"norad={norad}  metrique={metrique}  var_Q={var_Q:.4g} r={r:.4g} p0={p0:.4g} F1={prf['f1']:.3f} "
          f"P={prf['precision']:.3f} R={prf['recall']:.3f}  {conf}")
    return {"var_Q": var_Q, "r": r, "p0": p0, **prf, **conf}
