
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_filter_ordre_1(df, var_Q = 0.13, r = 1.0 , p0= 1000.0, metrique = "sma"):

    #Récupération des données : demi-grand-axe des tle du dataframe
    a = df[metrique].to_numpy().astype(float)
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

        y = f.y.item() 
        S = f.S.item() # H P^-1 H^T  + R
        residuals.append(y)
        normalized_residuals.append( y * y / S) # scalaire, suit une loi du chi2 à dimZ=1 degré de liberté
        sma_dot.append(f.x[1])              # dérivée du demi-grand axe estimée [km/jour]

    ## print("Filtre Kalman 1 appliqué...")

    return residuals, normalized_residuals, np.asarray(sma_dot)

def kalman_filter_ordre_2(df, var_Q = 0.13, r = 1.0, p0 = 1000.0, metrique='sma'):

    #Récupération des données : demi-grand-axe des tle du dataframe
    a = df['sma'].to_numpy().astype(float)
    epoch = df['epoch'].to_numpy()
    dt = (np.diff(epoch)/np.timedelta64(1,"D")).astype(float)
    n = len(a)

    # Initialisation du filtre Kalman 
    f = KalmanFilter(dim_x=3, dim_z=1)

    a_dot_0 = (a[1] - a[0])/dt[0]
    a_dot_1 = (a[2] - a[1])/dt[1]

    f.x = np.array([a[0],a_dot_0, (a_dot_1 - a_dot_0)/((dt[0]+dt[1])/2)])
    f.H = np.array([[1., 0. , 0.]]) # dimension dim_z = 1 * dim_x= 3
    f.P = np.array([[p0, 0., 0. ],
                    [0., p0, 0.],  # dimension dim_x ^2 
                    [0., 0., p0] ])
    f.R = np.array([[r]]) # r = 1.0 km^2 par défaut = bruit de mesure sur le semi-grand axe

    residuals = []
    normalized_residuals = []
    sma_dot = []                            # estimation filtrée de da/dt à chaque pas

    for k in range(1, n): 

        dt_k = dt[k-1]  
        F_k = np.array([[1., dt_k, (dt_k**2) / 2], 
                        [0., 1., dt_k],
                        [0., 0., 1.]])
        
        Q_k = Q_discrete_white_noise(dim=3, dt= dt_k , var=var_Q)

        f.predict(F= F_k , Q = Q_k)

        f.update(a[k])
        y = f.y.item() # résidus non normalisés 
        S = f.S.item() # H P^-1 H^T  + R
        residuals.append(y)
        normalized_residuals.append( y * y / S) # suit une loi du chi2 à dimZ=1 deg de liberté
        sma_dot.append(f.x[1])              # dérivée du demi-grand axe estimée [km/jour]

    ##print("Filtre Kalman 2 appliqué...")

    return residuals, normalized_residuals, np.asarray(sma_dot)

def detect_kalman(df, ordre = 1, var_Q=0.13, r=1.0, p0=1000.0, alpha=0.997, plot=True, ax=None, xlim=None, metrique='sma', tol_days=1.0):
    if ordre == 1 :
        _, nis, sma_dot = kalman_filter_ordre_1(df, var_Q=var_Q, r=r, p0=p0, metrique=metrique)
    elif ordre == 2 :
        _, nis, sma_dot = kalman_filter_ordre_2(df, var_Q=var_Q, r=r, p0=p0, metrique=metrique)
    else :
        raise Exception("choose 1 or 2 for LKF order")
    
    nis = np.asarray(nis, dtype=float)
    sma_dot = np.asarray(sma_dot, dtype=float)

    threshold = chi2.ppf(alpha, df=1)
    
    maneuvers = np.where(nis > threshold)[0] ## tableau de dates de manoeuvres prédites
    print(f"prise en compte de l'imprécision : tol_days={tol_days}")
    

    epoch = df['epoch'].to_numpy()[1:]          # séries alignées sur les pas k >= 1
    # Bornes x : mêmes que plot_time_series (epoch complet, pas [1:]) pour aligner
    # parfaitement les deux figures. `xlim` peut être passé identique aux deux.
    epoch_full = df['epoch'].to_numpy()
    xlim = xlim if xlim is not None else (epoch_full[0], epoch_full[-1])

    if maneuvers.size > 0 : 
        man_with_tol = []

        man_with_tol.append(maneuvers[0]) ## on prend la première man

        for i in range(1, len(maneuvers)):
            deltat = (epoch[maneuvers[i]] - epoch[maneuvers[i-1]]) / np.timedelta64(1, 'D')
            if deltat > tol_days : 
                man_with_tol.append(maneuvers[i])

        man_with_tol = np.asarray(man_with_tol)
    
    else :
        man_with_tol = maneuvers
    
    if ax is not None:
        # Mode comparaison : panneau unique avec nis 
        ax.plot(epoch, nis, lw=0.8, color="tab:blue")
        ax.set_ylabel(r"NIS $= y^2/S$")
        ax.set_xlabel("epoch")
        ax.set_title(r"Kalman ($\varepsilon$)")
        if len(man_with_tol):
            ax.scatter(epoch[man_with_tol], nis[man_with_tol], color="red",
                       marker="x", zorder=5, label=f"manœuvres ({len(man_with_tol)})")
            for m in man_with_tol:
                ax.axvline(epoch[m], color="red", ls=":", lw=0.6, alpha=0.5)
            ax.legend(loc="best", fontsize=8)
        ax.set_xlim(xlim)
        ax.margins(x=0)
        return epoch, sma_dot, nis, man_with_tol, threshold

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # dérivée estimée du demi-grand axe (saute à chaque manœuvre)
        ax1.plot(epoch, sma_dot, lw=0.8, color="tab:blue")
        ax1.set_ylabel(fr"$ \Delta {metrique} $ estimé [km/jour]")

        # Statistique de détection : NIS comparée au seuil χ²
        ax2.plot(epoch, nis, lw=0.8, label="NIS")
        ax2.axhline(threshold, color="grey", ls="--", lw=0.8,
                    label=rf"seuil $\chi^2_1$($\alpha$={alpha}) = {threshold:.2f}")
        ax2.set_ylabel(r"NIS $= y^2/S$")
        ax2.set_xlabel("epoch")
        ax2.legend(loc="best", fontsize=8)

        if len(man_with_tol):
            for ax, sig in ((ax1, sma_dot), (ax2, nis)):
                ax.scatter(epoch[man_with_tol], sig[man_with_tol], color="red",
                           marker="x", zorder=5,
                           label=f"manœuvres ({len(man_with_tol)})")
                for m in man_with_tol:
                    ax.axvline(epoch[m], color="red", ls=":", lw=0.6, alpha=0.5)
            ax1.legend(loc="best", fontsize=8)

        # sharex=True -> régler ax2 suffit ; bornes identiques à plot_time_series
        ax2.set_xlim(xlim)
        ax2.margins(x=0)
        plt.tight_layout()
        plt.show()

    return epoch, sma_dot, nis, man_with_tol, threshold
