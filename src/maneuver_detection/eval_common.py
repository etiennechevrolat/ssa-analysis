"""Convention d'evaluation commune aux deux methodes de detection.

Le depot contenait trois appariements incompatibles :

  - src/optimisation_seuil/metrics.py:4      non exclusif, tol_days = 1 a 2 j
  - sso_annotation/scripts/03_compare.py:33  1-1 glouton, tol = 1.5 j
  - src/ml/evaluate.py:69                    1-1, tolerance en pas de temps

Un appariement NON EXCLUSIF gonfle le rappel : une seule detection posee au milieu
d'une salve de manoeuvres vraies les valide toutes. Comparer Kalman et le test
statistique avec deux conventions differentes ne veut rien dire, donc tout passe
ici : 1-1 glouton au plus proche, tolerance en jours.
"""

import numpy as np

TOLERANCE_J = 1.5


def apparie(t_cand, t_vrai, tol=TOLERANCE_J):
    """Appariement 1-1 au plus proche. Retourne (tp, fp, fn).

    Identique a sso_annotation/scripts/03_compare.py:33, recopie ici pour que la
    chaine src/ ne depende pas d'un script non packageable.
    """
    t_cand = np.sort(np.asarray(t_cand, dtype=float))
    t_vrai = np.sort(np.asarray(t_vrai, dtype=float))
    pris = np.zeros(len(t_vrai), bool)
    tp = 0
    for tc in t_cand:
        libres = np.flatnonzero(~pris)
        if not len(libres):
            continue
        j = libres[np.argmin(np.abs(t_vrai[libres] - tc))]
        if abs(t_vrai[j] - tc) <= tol:
            pris[j] = True
            tp += 1
    return tp, len(t_cand) - tp, int((~pris).sum())


def fbeta(beta, tp, fp, fn):
    """F-beta. beta > 1 privilegie le rappel (F2), beta < 1 la precision."""
    d = (1 + beta ** 2) * tp + (beta ** 2) * fn + fp
    return (1 + beta ** 2) * tp / d if d > 0 else 0.0


def metriques(tp, fp, fn, duree_objet_an=None):
    """TP/FP/FN -> precision, rappel, F1, F2 (+ fausses alarmes par objet-an)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    rappel = tp / (tp + fn) if (tp + fn) else 0.0
    out = {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "rappel": rappel,
        "f1": fbeta(1.0, tp, fp, fn), "f2": fbeta(2.0, tp, fp, fn),
    }
    if duree_objet_an:
        out["fa_par_objet_an"] = fp / duree_objet_an
    return out


def agrege(lignes, duree_objet_an=None):
    """Micro-moyenne : on somme les comptages avant de calculer les taux.

    Une macro-moyenne des F1 par objet donnerait le meme poids a un objet couvert
    par 3 TLE et a un objet couvert par 30 000.
    """
    tp = sum(l["tp"] for l in lignes)
    fp = sum(l["fp"] for l in lignes)
    fn = sum(l["fn"] for l in lignes)
    if duree_objet_an is None:
        duree_objet_an = sum(l.get("duree_an", 0.0) for l in lignes) or None
    return metriques(tp, fp, fn, duree_objet_an)


def to_days(times, t0):
    """datetime64 -> jours flottants depuis t0."""
    arr = np.asarray(times, dtype="datetime64[ns]")
    return (arr - np.datetime64(t0, "ns")).astype("float64") / 8.64e13


def restreint_fenetre(t_vrai_d, t_serie_d, marge=0.0):
    """Manoeuvres tombant dans la couverture temporelle des TLE.

    optimise_seuil.py ne le fait PAS quand man_ilrs=False : toute manoeuvre
    labellisee hors couverture TLE compte alors en FN, ce qui plafonne le rappel
    a une valeur qui ne depend pas du detecteur.
    """
    if len(t_serie_d) == 0:
        return t_vrai_d[:0]
    lo, hi = t_serie_d[0] - marge, t_serie_d[-1] + marge
    return t_vrai_d[(t_vrai_d >= lo) & (t_vrai_d <= hi)]
