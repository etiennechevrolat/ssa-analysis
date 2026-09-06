"""Detecteurs de rupture sur series de parametres orbitaux.

Rapatrie depuis sso_annotation/scripts/detecteurs.py, supprime du depot avec le
reste de sso_annotation/ (506 Mo de caches et de planches). Seul le code est
conserve ici : les scripts d'evaluation et de figures en dependent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

GM = 398600.4418          # km^3/s^2

# --------------------------------------------------------------------------- preparation

def mad(v):
    v = np.asarray(v, float)
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-12


def bruit_hf(v):
    """Ecart-type du bruit haute frequence : var(x[t]-2x[t-1]+x[t-2]) = 6 sigma^2.

    Robuste par le MAD, donc insensible aux marches elles-memes -- une marche ne touche
    que trois termes de la difference seconde.
    """
    return mad(np.diff(np.asarray(v, float), 2)) / np.sqrt(6)


def nettoie(df, n_sigma=8.0):
    """Retire les TLE ponctuellement aberrants, et EUX SEULS.

    Un outlier isole s'ecarte de l'interpolation de ses deux voisins ; une marche de
    manoeuvre, elle, deplace tout ce qui suit, donc ses voisins ne l'encadrent pas.
    Le test porte donc sur |a[k] - (a[k-1]+a[k+1])/2|, et l'echelle vient de la
    difference SECONDE (bruit_hf), jamais du MAD du residu : les TLE republies a valeur
    quasi identique rendent ce residu nul pour la moitie des points, le MAD s'effondre et
    le seuil avec lui -- ce qui coupait 26 % de la serie et divisait le bruit par quinze.
    """
    a = df.sma.to_numpy(float)
    if len(a) < 11:
        return df
    sigma = bruit_hf(a)
    ecart = np.full(len(a), 0.0)
    ecart[1:-1] = a[1:-1] - 0.5 * (a[:-2] + a[2:])
    ## sqrt(1.5) : variance de a_k - (a_{k-1}+a_{k+1})/2 pour un bruit blanc d'ecart-type sigma
    return df[np.abs(ecart) < n_sigma * sigma * np.sqrt(1.5)].copy()


def serie(df):
    """(t en jours depuis le debut, sma en m, inclinaison en deg, epochs) -- serie nettoyee."""
    df = df.sort_values("epoch")
    df = nettoie(df)
    ep = pd.to_datetime(df.epoch).dt.tz_localize(None).to_numpy("datetime64[ns]")
    t = (ep - ep[0]).astype("float64") / 86400e9
    return t, df.sma.to_numpy(float) * 1000.0, df.inclination.to_numpy(float), ep, df


# --------------------------------------------------------------------------- utilitaires

def sommets(score, seuil, separation):
    """Indices des maxima locaux de `score` au-dessus de `seuil`, separes d'au moins
    `separation` echantillons -- suppression des non-maxima, en une passe gloutonne du
    plus fort au plus faible (le meme principe que la NMS des detecteurs d'objets)."""
    cand = np.flatnonzero(score > seuil)
    if not len(cand):
        return np.array([], int)
    cand = cand[np.argsort(-score[cand])]
    gardes = []
    for i in cand:
        if all(abs(i - j) >= separation for j in gardes):
            gardes.append(i)
    return np.array(sorted(gardes), int)


def _amplitude(t, y, i, demi):
    """Marche en i, estimee par deux regressions lineaires encadrant la rupture.

    Extrapolees toutes deux a l'instant de rupture : c'est ce qui distingue une marche
    d'un changement de PENTE (une manoeuvre change le niveau, la trainee change la pente).
    Renvoie (marche, erreur type de la marche, saut de pente).
    """
    g = slice(max(0, i - demi), i)
    d = slice(i + 1, min(len(y), i + 1 + demi))
    if (g.stop - g.start) < 5 or (d.stop - d.start) < 5:
        return np.nan, np.nan, np.nan
    tc = 0.5 * (t[i] + t[min(i + 1, len(t) - 1)])
    out = []
    for sl in (g, d):
        A = np.vstack([t[sl] - tc, np.ones(t[sl].shape)]).T
        beta, *_ = np.linalg.lstsq(A, y[sl], rcond=None)
        res = y[sl] - A @ beta
        ddl = max(len(res) - 2, 1)
        cov = np.linalg.pinv(A.T @ A) * (res @ res) / ddl
        out.append((beta, cov))
    (bg, cg), (bd, cd) = out
    marche = bd[1] - bg[1]
    err = np.sqrt(cg[1, 1] + cd[1, 1]) + 1e-9
    return marche, err, bd[0] - bg[0]


# --------------------------------------------------------------------------- D1

def d1_pas_median(t, y, demi=10, seuil=6.0, separation=8):
    """Mediane des `demi` points avant vs apres, z robuste sur la distribution des pas."""
    n = len(y)
    if n < 4 * demi:
        return pd.DataFrame(columns=["i", "t", "amplitude", "score"])
    pas = np.full(n, np.nan)
    for i in range(demi, n - demi):
        pas[i] = np.median(y[i + 1:i + 1 + demi]) - np.median(y[i - demi:i])
    fini = np.isfinite(pas)
    z = np.zeros(n)
    z[fini] = np.abs(pas[fini] - np.median(pas[fini])) / mad(pas[fini])
    idx = sommets(z, seuil, separation)
    return pd.DataFrame({"i": idx, "t": t[idx], "amplitude": pas[idx], "score": z[idx]})


# --------------------------------------------------------------------------- D2

def pas_quantification(y):
    """Plus petit ecart non nul entre valeurs consecutives : le quantum d'ecriture TLE.

    L'inclinaison n'a que 4 decimales dans un TLE, donc 1e-4 deg. Sur un satellite calme
    la serie reste plusieurs points sur le MEME palier : la variance residuelle locale
    tombe a zero et toute t-statistique explose (mesure : t = 1e5 sur un saut d'un seul
    quantum). Le pas rendu ici sert de plancher a l'ecart type residuel.
    """
    d = np.abs(np.diff(np.asarray(y, float)))
    d = d[d > 0]
    return float(np.min(d)) if len(d) else 0.0


def d2_filtre_adapte(t, y, demi=12, seuil=8.0, separation=8, ordre=1, avec_kink=True,
                     retourne_courbe=False):
    """Filtre adapte a une marche, teste CONTRE les alternatives continues.

    Sur une fenetre centree en i on ajuste en une seule regression :

        y = c0 + c1 tau + c2 tau^2 + A * 1[tau > 0] + B * max(tau, 0)

    et le detecteur est la t-statistique de A.

      - ordre=1 par defaut : avec B au modele, la droite brisee absorbe deja la courbure
        de la decroissance par trainee. Ajouter tau^2 par-dessus rend la fenetre de 24
        points mal conditionnee (tau^2, 1[tau>0] et max(tau,0) deviennent colineaires),
        la pseudo-inverse sous-estime la variance de A et le taux de fausse alarme sur
        les temoins TRIPLE (0.045 -> 0.119 par objet-an) a rappel egal ;
      - le terme B absorbe un CHANGEMENT DE PENTE continu -- une bouffee d'activite solaire,
        un changement d'attitude qui modifie le maitre-couple. C'est le principal faux
        positif restant sur les debris : la serie ne saute pas, elle plonge plus vite.
        Avec B au modele, seule une vraie DISCONTINUITE alimente A ;
      - l'ecart type residuel est plancher par le pas de quantification / sqrt(12) : sinon
        une serie posee sur un palier donne une erreur type nulle et un t infini.

    Le seuil porte sur une grandeur sans dimension, donc transposable d'un satellite a
    l'autre sans reglage -- contrairement a un seuil en metres.
    """
    
    n = len(y)
    if n < 4 * demi:
        return np.zeros(n) if retourne_courbe else pd.DataFrame(
            columns=["i", "t", "amplitude", "score", "d_pente"])
    var_plancher = (pas_quantification(y) ** 2) / 12.0
    marche = np.full(n, np.nan)
    dpente = np.full(n, np.nan)
    tstat = np.zeros(n)
    for i in range(demi, n - demi):
        sl = slice(i - demi + 1, i + demi + 1)
        tau = t[sl] - 0.5 * (t[i] + t[i + 1])
        cols = [np.ones_like(tau)] + [tau ** k for k in range(1, ordre + 1)]
        cols.append((tau > 0).astype(float))
        if avec_kink:
            cols.append(np.maximum(tau, 0.0))
        A = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(A, y[sl], rcond=None)
        res = y[sl] - A @ beta
        ddl = len(res) - A.shape[1]
        if ddl <= 1:
            continue
        var = max((res @ res) / ddl, var_plancher)
        cov = np.linalg.pinv(A.T @ A) * var
        j = ordre + 1                                   # indice du coefficient de marche
        err = np.sqrt(max(cov[j, j], 1e-18))
        marche[i], tstat[i] = beta[j], abs(beta[j]) / err
        dpente[i] = beta[-1] if avec_kink else np.nan
    if retourne_courbe:
        return tstat
    idx = sommets(tstat, seuil, separation)
    return pd.DataFrame({"i": idx, "t": t[idx], "amplitude": marche[idx],
                         "score": tstat[idx], "d_pente": dpente[idx]})


# --------------------------------------------------------------------------- D3

def _gain_bruit(t, grille, sigma_grille, echelle, n_tirages=8, graine=0):
    """Ecart-type de la reponse du filtre a du bruit blanc de variance 1 sur la serie.

    Calibration necessaire parce que la normalisation ne peut PAS venir du MAD de la
    reponse : a grande echelle celle-ci est dominee par les variations lentes reelles
    (trainee, activite solaire), pas par le bruit -- le MAD y vaut quinze fois sa valeur a
    petite echelle et le seuil devient inatteignable. On mesure donc le gain par simulation,
    en propageant du bruit blanc a travers l'interpolation ET le filtre : c'est la
    normalisation CFAR du detecteur.
    """
    rng = np.random.default_rng(graine)
    ecarts = []
    for _ in range(n_tirages):
        b = np.interp(grille, t, rng.standard_normal(len(t)))
        d = gaussian_filter1d(b, sigma=sigma_grille, order=1) * sigma_grille
        ecarts.append(np.std(d))
    return float(np.mean(ecarts))


def d3_canny(t, y, echelles=(0.25, 0.5, 1.0, 2.0), pas_grille=0.25, seuil_haut=6.0,
             seuil_bas=3.0, persistance=2, separation_j=3.0, fond_j=20.0):
    """Detecteur de contours de Canny, applique a la serie vue comme une image 1D.

    Etapes canoniques : derivee de gaussienne (le detecteur de marche optimal de Canny),
    normalisation par le bruit, suppression des non-maxima, hysteresis. Trois adaptations :

      - reechantillonnage sur grille reguliere : les operateurs de convolution supposent
        un pas constant, les TLE arrivent a intervalles irreguliers ;
      - SOUSTRACTION DE FOND par mediane glissante sur la reponse (chapeau haut-de-forme
        morphologique) : la decroissance par trainee donne a la derivee un plateau non nul
        qui noie le pic de la marche -- un fond d'image inhomogene, exactement ;
      - normalisation par _gain_bruit et non par la dispersion de la reponse (cf. supra).

    Persistance en echelle (edge focusing) : la reponse doit depasser le seuil haut a au
    moins `persistance` echelles. Un pic de bruit est fort a une echelle et disparait aux
    autres, une vraie marche survit.
    """
    if len(t) < 40:
        return pd.DataFrame(columns=["i", "t", "amplitude", "score"])
    grille = np.arange(t[0], t[-1], pas_grille)
    yg = np.interp(grille, t, y)
    sigma_bruit = bruit_hf(y)
    n_fond = max(int(fond_j / pas_grille) | 1, 5)

    reponses = []
    for s in echelles:
        sg = s / pas_grille
        d = gaussian_filter1d(yg, sigma=sg, order=1) * sg
        fond = pd.Series(d).rolling(n_fond, center=True, min_periods=1).median().to_numpy()
        r = d - fond
        reponses.append(np.abs(r) / (sigma_bruit * _gain_bruit(t, grille, sg, s) + 1e-12))
    R = np.vstack(reponses)

    forts = (R > seuil_haut).sum(axis=0) >= persistance
    score = R.max(axis=0)
    idx_g = sommets(score * (score > seuil_bas), seuil_bas, int(separation_j / pas_grille))
    idx_g = [i for i in idx_g if forts[i]]                       # hysteresis : germe fort

    lignes = []
    for ig in idx_g:
        i = int(np.searchsorted(t, grille[ig]))
        i = min(max(i, 1), len(t) - 2)
        m, _, _ = _amplitude(t, y, i, 12)
        lignes.append({"i": i, "t": t[i], "amplitude": m, "score": float(score[ig])})
    return pd.DataFrame(lignes, columns=["i", "t", "amplitude", "score"])


# --------------------------------------------------------------------------- D4

def d4_kalman(t, y, var_q=0.13, r=None, p0=1000.0, alpha=0.997, separation_j=3.0):
    """Pics de l'innovation normalisee d'un Kalman ordre 1 (position + derive).

    Reprise de src/maneuver_detection/discrete_kalman_filter.py, reecrite ici pour rester
    autonome et pour que R soit ajuste sur le bruit mesure de l'objet : le r=1.0 km^2 par
    defaut du code d'origine vaut 1e6 m^2, six ordres de grandeur au-dessus du bruit TLE
    reel (~3 m), ce qui rend le filtre sourd a toute manoeuvre.
    """
    from filterpy.common import Q_discrete_white_noise
    from filterpy.kalman import KalmanFilter

    n = len(y)
    if n < 20:
        return pd.DataFrame(columns=["i", "t", "amplitude", "score"])
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    r = (bruit_hf(y) ** 2) if r is None else r

    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([y[0], (y[1] - y[0]) / dt[0]])
    f.H = np.array([[1.0, 0.0]])
    f.P = np.array([[p0, 0.0], [0.0, p0]])
    f.R = np.array([[r]])

    nis = np.zeros(n)
    for k in range(1, n):
        dtk = dt[k - 1]
        f.predict(F=np.array([[1.0, dtk], [0.0, 1.0]]),
                  Q=Q_discrete_white_noise(dim=2, dt=dtk, var=var_q))
        f.update(y[k])
        nis[k] = f.y.item() ** 2 / f.S.item()

    seuil = chi2.ppf(alpha, df=1)
    sep = max(int(np.round(separation_j / np.median(dt))), 1)
    idx = sommets(nis, seuil, sep)
    lignes = []
    for i in idx:
        m, _, _ = _amplitude(t, y, int(i), 12)
        lignes.append({"i": int(i), "t": t[i], "amplitude": m, "score": nis[i]})
    return pd.DataFrame(lignes, columns=["i", "t", "amplitude", "score"])


# --------------------------------------------------------------------------- Delta v

def dv_in_track(a_m, da_m):
    """|Dv| tangentiel produisant une variation da du demi-grand axe (orbite quasi circulaire).

    v = sqrt(GM/a) et dv = (v/2)(da/a) : c'est la derivee de vis-viva a r fixe.
    """
    v = np.sqrt(GM / (a_m / 1000.0)) * 1000.0            # m/s
    return np.abs(0.5 * v * da_m / a_m)


def dv_cross_track(a_m, di_deg):
    """|Dv| hors plan produisant une variation di de l'inclinaison : dv = v * di."""
    v = np.sqrt(GM / (a_m / 1000.0)) * 1000.0
    return np.abs(v * np.radians(di_deg))


def dv_total(a_m, da_m, di_deg):
    """Composition des deux composantes, orthogonales par construction."""
    return np.hypot(dv_in_track(a_m, da_m), dv_cross_track(a_m, di_deg))
