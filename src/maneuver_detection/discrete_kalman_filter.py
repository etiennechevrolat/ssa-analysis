
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_filter(df, var_Q = 0.13, r = 1.0 , p0= 1000.0):
    #Récupération des données : demi-grand-axe des tle du dataframe
    a = df['sma'].to_numpy().astype(float)
    epoch = df['epoch'].to_numpy()
    dt = (np.diff(epoch)/np.timedelta64(1,"D")).astype(float)
    n = len(a)

    # Initialisation du filtre Kalman 
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([a[0],(a[1] - a[0])/dt[0]])
    f.H = np.array([[1., 0.]])
    f.P = np.array([[p0,    0.],
                    [   0., p0] ])
    f.R = np.array([[r]]) # r = 1.0 km^2 par défaut = bruit de mesure sur le semi-grand axe

    

    residuals = []
    normalized_residuals = []
    sma_dot = []                            # estimation filtrée de da/dt à chaque pas

    for k in range(1, n): 

        dt_k = dt[k-1]  
        F_k = np.array([[1., dt_k], 
                        [0., 1.]])
        Q_k = Q_discrete_white_noise(dim=2, dt= dt_k , var=var_Q)
        
        f.predict(F= F_k , Q = Q_k)
        f.update(a[k])

        y = f.y.item() # scalaire, suit une loi du chi2 à dimZ=1 degré de liberté
        S = f.S.item() # H P^-1 H^T  + R
        residuals.append(y)
        normalized_residuals.append( y * y / S)
        sma_dot.append(f.x[1])              # dérivée du demi-grand axe estimée [km/jour]

    print("Filtre Kalman appliqué...")

    return residuals, normalized_residuals, np.asarray(sma_dot)


def detect_kalman(df, var_Q=0.13, r=1.0, p0=1000.0, alpha=0.997, plot=True, ax=None):

    _, nis, sma_dot = kalman_filter(df, var_Q=var_Q, r=r, p0=p0)
    nis = np.asarray(nis, dtype=float)
    sma_dot = np.asarray(sma_dot, dtype=float)

    threshold = chi2.ppf(alpha, df=1)
    maneuvers = np.where(nis > threshold)[0]

    epoch = df['epoch'].to_numpy()[1:]          # séries alignées sur les pas k >= 1

    if ax is not None:
        # Mode comparaison : panneau unique avec la dérivée da/dt (démo Zollo)
        ax.plot(epoch, sma_dot, lw=0.8, color="tab:blue")
        ax.set_ylabel(r"$\dot a$ estimé [km/jour]")
        ax.set_xlabel("epoch")
        ax.set_title("Kalman (da/dt)")
        if len(maneuvers):
            ax.scatter(epoch[maneuvers], sma_dot[maneuvers], color="red",
                       marker="x", zorder=5, label=f"manœuvres ({len(maneuvers)})")
            for m in maneuvers:
                ax.axvline(epoch[m], color="red", ls=":", lw=0.6, alpha=0.5)
            ax.legend(loc="best", fontsize=8)
        return epoch, sma_dot, nis, maneuvers, threshold

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # dérivée estimée du demi-grand axe (saute à chaque manœuvre)
        ax1.plot(epoch, sma_dot, lw=0.8, color="tab:blue")
        ax1.set_ylabel(r"$\dot a$ estimé [km/jour]")

        # Statistique de détection : NIS comparée au seuil χ²
        ax2.plot(epoch, nis, lw=0.8, label="NIS")
        ax2.axhline(threshold, color="grey", ls="--", lw=0.8,
                    label=rf"seuil $\chi^2_1$($\alpha$={alpha}) = {threshold:.2f}")
        ax2.set_ylabel(r"NIS $= y^2/S$")
        ax2.set_xlabel("epoch")
        ax2.legend(loc="best", fontsize=8)

        if len(maneuvers):
            for ax, sig in ((ax1, sma_dot), (ax2, nis)):
                ax.scatter(epoch[maneuvers], sig[maneuvers], color="red",
                           marker="x", zorder=5,
                           label=f"manœuvres ({len(maneuvers)})")
                for m in maneuvers:
                    ax.axvline(epoch[m], color="red", ls=":", lw=0.6, alpha=0.5)
            ax1.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.show()

    return epoch, sma_dot, nis, maneuvers, threshold
