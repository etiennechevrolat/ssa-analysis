from scipy.optimize import minimize as scipy_minimize
import numpy as np

def confusion_matrix(maneuvers, predictions, tol_days=1.0):
    """
    Compute the confusion matrix for maneuver detection.
    maneuvers / predictions : temps EN JOURS (float), même origine.
    tol_days : fenêtre d'appariement (une prédiction compte si elle tombe
               à moins de tol_days d'une manœuvre vraie).

    return a dictionary with the following keys:
        -n_miss : number of maneuvers not detected
        -n_false : number of false positives
        -n_det : number of maneuvers correctly detected
    """
    true_t = np.asarray(maneuvers, dtype=float)
    pred_t = np.asarray(predictions, dtype=float)

    n_miss = sum(
        not np.any(np.abs(pred_t - m) < tol_days) for m in true_t
    )
    n_false = sum(
        not np.any(np.abs(true_t - p) < tol_days) for p in pred_t
    )
    n_det = len(true_t) - n_miss

    return {"n_miss": n_miss, "n_false": n_false, "n_det": n_det}

def precision_recall_f1(confusion):
    
    n_miss = confusion["n_miss"]
    n_false = confusion["n_false"]
    n_det = confusion["n_det"]

    precision = n_det / (n_det + n_false) if (n_det + n_false) > 0 else 0
    recall = n_det / (n_det + n_miss) if (n_det + n_miss) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

def lissage_noyau_gaussien_metriques(maneuvers, predictions, scores, seuil, sigma, beta):
    true_maneuvers =  np.asarray(maneuvers, dtype=float)[:,None] #(N,1)
    predicted_maneuvers = np.asarray(predictions, dtype=float)[None, :] #(1,M)
    s = np.asarray(scores, dtype=float)

    w = np.exp(-(true_maneuvers - predicted_maneuvers)**2 / (2*sigma**2)) # (N,M) poids de correspondance tempo
    p = 1.0 / (1.0 + np.exp(-(scores - seuil) / beta))

    recall = np.mean(np.max(w*p[None, :], axis=1))
    denom = p.sum()
    precision = (p * w.max(axis=0)).sum() / denom if denom > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return 1.0 - f1

def minimize(loss_fn, x0, bounds=None):
    """Minimise loss_fn, en pratique notre loss F1 lissée par Powell
    """
    return scipy_minimize(loss_fn, x0, method="Powell", bounds=bounds)
