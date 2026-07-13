import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks


def detect_peaks(residual, t=None, threshold_sigma=4.0, min_distance_days=None, prominence=None):

    x = np.abs(np.asarray(residual, dtype=float))
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return np.array([], dtype=int), np.nan

    mean = np.mean(x[finite])
    var = np.mean(np.abs(x[finite] - mean))

    threshold = mean + threshold_sigma * var 

    # Distance minimale convertie de jours en nombre d'échantillons
    distance = None
    if min_distance_days is not None and t is not None:
        t = np.asarray(t, dtype=float)
        if t.size > 1:
            dt_mean = np.mean(np.diff(t))
            if dt_mean > 0:
                distance = max(1, int(round(min_distance_days / dt_mean)))

    # find_peaks n'accepte pas les NaN : on les neutralise à 0
    xf = np.where(finite, x, 0.0)
    peaks, _ = find_peaks(xf, height=threshold, distance=distance, prominence=prominence)
    return peaks, threshold


def _mark_maneuvers(ax, x, y, peaks, threshold=None):
    """Surligne sur ``ax`` les manoeuvres détectées (indices ``peaks``)."""
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color="grey", ls="--", lw=0.7,
                   label="seuil de détection")
        ax.axhline(-threshold, color="grey", ls="--", lw=0.7)
    
    if len(peaks):
        ax.scatter(np.asarray(x)[peaks], np.asarray(y)[peaks],
                   color="red", marker="x", zorder=5,
                   label=f"manoeuvres ({len(peaks)})")
        for p in peaks:
            ax.axvline(x[p], color="red", ls=":", lw=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=8)


def MLF_LOWESS_KEPLERIAN(df, smoothed=True, detect=True, threshold_sigma=4.0,
                         min_distance_days=7.0, ax=None):

    epoch = df['epoch'].to_numpy()
    n_rev_day = df['mean_motion'].to_numpy().astype(float)

    # longitude moyenne (deg) non singulière lorsque e, i -> 0
    lbda = (df['raan'] + df['arg_perigee'] + df['mean_anomaly']).to_numpy().astype(float) % 360
    n = len(epoch)

    ## durées (jours) entre deux enregistrement consécutifs
    conversion = np.timedelta64(1, "D")
    dt = (np.diff(epoch) / conversion).astype(float)
    
    # Longitude moyenne (deg) propagée par equation keplerienne
    lbda_propag = ((lbda[:-1] + n_rev_day[:-1]*360*dt) % 360)

    # Différences résiduelles (deg) en long moyenne
    diff = lbda[1:] - lbda_propag
    # repli sur [-180, 180] pour éviter les sauts de 360 deg parasites
    diff = (diff + 180) % 360 - 180

    t = (epoch[1:] - epoch[0]) / conversion.astype(float)

    if not smoothed:
        signal = diff
        ylabel = "Résidus en longitude moyenne [deg]"
    else:
        ## Filtration par smoothing LOWESS
        signal = lowess(diff, t, frac=0.008, return_sorted=False)
        ylabel = r"$\Delta\lambda$ (résidu Keplerien) [deg]"

    peaks, threshold = (detect_peaks(signal, t, threshold_sigma, min_distance_days)
                        if detect else (np.array([], dtype=int), np.nan))
    
    if ax is not None:
        # Mode comparaison : on trace sur l'axe fourni, sans ouvrir de figure
        ax.plot(epoch[1:], signal, lw=0.8)
        ax.set_xlabel("epoch (days)")
        ax.set_ylabel(ylabel)
        
        if detect:
            _mark_maneuvers(ax, epoch[1:], signal, peaks, threshold)
        return signal, peaks

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(epoch[1:], signal, lw=0.8)
    ax.set_xlabel("epoch (days)")
    ax.set_ylabel(ylabel)
    if detect:
        _mark_maneuvers(ax, epoch[1:], signal, peaks, threshold)
    plt.show()

    return signal, peaks


from sgp4.api import Satrec, WGS72, jday


def MLF_LOWESS_SGP4(df, norad, detect=True, threshold_sigma=4.0,
                    min_distance_days=7.0, ax=None):
    N        = df.height
    epochs   = df['epoch'].to_list()        # datetime.datetime (UTC)
    rows     = df.to_dicts()
    DEG2RAD  = np.pi / 180.0
    TWO_PI   = 2 * np.pi

    def build_satrec(r, dt):
        jd, fr = jday(dt.year, dt.month, dt.day,
                    dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
        sat = Satrec()
        sat.sgp4init(
            WGS72, 'i', norad,
            (jd + fr) - 2433281.5,                 # epoch : jours depuis 1949-12-31 00:00 UT
            0.0,
            0.0, 0.0,                              # ndot, nddot ~ 0 en GEO
            r['eccentricity'],
            r['arg_perigee']  * DEG2RAD,           # argpo  [rad]
            r['inclination']  * DEG2RAD,           # inclo  [rad]
            r['mean_anomaly'] * DEG2RAD,           # mo     [rad]
            r['mean_motion']  * TWO_PI / 1440.0,   # no_kozai : rev/jour -> rad/min
            r['raan']         * DEG2RAD,           # nodeo  [rad]
        )
        return sat, jd, fr
    sats, jds, frs = [], [], []
    for r, dt in zip(rows, epochs):
        s, jd, fr = build_satrec(r, dt)
        sats.append(s); jds.append(jd); frs.append(fr)

    # Résidu de longitude inertielle, propagé vs observé au même instant t_k
    def ra_at(sat, jd, fr):
        e, rvec, _ = sat.sgp4(jd, fr)
        return np.nan if e != 0 else np.arctan2(rvec[1], rvec[0])   # rad
    dlam = np.full(N - 1, np.nan)
    for k in range(1, N):
        ra_obs  = ra_at(sats[k],   jds[k], frs[k])     # TLE_k à son epoch
        ra_prop = ra_at(sats[k-1], jds[k], frs[k])     # TLE_{k-1} propagé jusqu'à t_k
        dlam[k-1] = (ra_obs - ra_prop + np.pi) % TWO_PI - np.pi

    conversion = np.timedelta64(1, "D")
    ep = df['epoch'].to_numpy()
    t  = ((ep[1:] - ep[0]) / conversion).astype(float)
    dlam_smoothed = lowess(dlam, t, frac=0.005, return_sorted=False)

    peaks, threshold = (detect_peaks(dlam_smoothed, t, threshold_sigma, min_distance_days)
                        if detect else (np.array([], dtype=int), np.nan))

    if ax is not None:
        # Mode comparaison : axe des epochs (datetime) partagé avec les
        # autres méthodes, et non le temps en jours utilisé pour le calcul
        ax.plot(ep[1:], dlam_smoothed, lw=0.6)
        ax.set_ylabel(r"$\Delta\lambda$ (résidu SGP4) [rads]")
        ax.set_xlabel("epoch")
        ax.set_title(f"NORAD {norad}")
        if detect:
            _mark_maneuvers(ax, ep[1:], dlam_smoothed, peaks, threshold)
        return dlam_smoothed, peaks

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, dlam_smoothed, lw=0.6)
    ax.set_ylabel(r"$\Delta\lambda$ (résidu SGP4) [rads]")
    ax.set_xlabel("jours")
    ax.set_title(f"NORAD {norad}")
    if detect:
        _mark_maneuvers(ax, t, dlam_smoothed, peaks, threshold)
    plt.show()

    return dlam_smoothed, peaks
