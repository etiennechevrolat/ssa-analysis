import numpy as np
from collections import defaultdict

from ml.targets import doris_types

## Paramètres d'évaluation

detection_treshold = 0.1 ## manoeuvre détectée pour proba_pred > treshold
matching_tolerance = 6 ## 6jours de tolérance pour détection de manoeuvres

## Noeuds de manoeuvres réelle
pol_nodes= ('ID', 'AD', 'IK')



### EVALUTATION DU LOCALIZER: 
# Nécessite une mise des prédictions d'un tableau de taille N par batch avec des logits -> à des événements ponctuels dans les deux directions


## Mise des labels au format pour tous les objets :  {oid : {'EW' : [], 'NS' : []}}
def gt_events_from_labels(labels, # DataFrame : ObjectID, Direction, Node, TimeIndex, Type
                          objects_ids,
                          nodes=pol_nodes
    ):
    out={}
    for oid in objects_ids:
        sub = labels[labels['ObjectID'] == oid]
        event = {}
        for direction in ('EW', 'NS'):
            t = sub[(sub['Direction'] == direction) & (sub['Node'].isin(nodes))]['TimeIndex']
            event[direction] = sorted(int(x) for x in t.to_numpy())
        out[oid] = event 
    return out 


## Mise des prédictions au format events pour UN objet {'EW' : [], 'NS' : []}
def _run_centers(mask : np.ndarray)-> list[int]:
    """Récupère un mask de type [F,T,T,T,F,T,T,F] et renvoie les indices des centres des suites de True
    """
    idx = np.flatnonzero(mask) # indices ou mask est true
    if idx.size == 0:
        return []
    # on coupe idx en différents runs, ie. la ou deux indices ne se suivent pas.
    runs = np.split(idx, np.flatnonzero(np.diff(idx)> 1) + 1)
    # on renvoie le milieu de ces différents runs
    return [run[len(run)//2]  for run in runs]

def extract_events(scores : np.ndarray, # tableau (L, n_canaux), une colonne par canal
                treshold: float | dict[str, float] = detection_treshold,
                is_logits: bool=False,
                channels : tuple[str, ...] = ('EW', 'NS')
    )-> dict[str, list[int]] :
    """(L, n_canaux) scores -> {canal : [t..]} après seuillage + collapse au centre.
    treshold accepte un flottant unique ou un seuil par canal {"EW": .., "NS": ..} :
    les manoeuvres in-plane et out-of-plane n'ont pas le même rapport signal/bruit, et
    un seuil commun dégrade la direction la moins bien séparée.
    channels nomme les colonnes : EW/NS pour SPLID, in-track / cross-track pour DORIS.
    """
    if is_logits:
        scores = 1/ (1 + np.exp(-scores))
    events = {}
    for col, name in enumerate(channels):
        th = treshold[name] if isinstance(treshold, dict) else treshold
        mask = scores[:, col] > th
        events[name] = _run_centers(mask)
    return events
 
## Matching entre les labels et prédictions 
def match_events(gt_idx : list[int],  # liste d'instants de vraies manoeuvres
                pred_idx: list[int],  # liste d'instants de manoeuvres prédites
                tolerance=matching_tolerance)-> dict:
    """Matche deux listes triées d'indices selon UNE direction pour UN objet)."""

    matched = set() # indices de pred_idx déjà appariés
    tp, fn = 0,0
    distances= []
    ## Les INSTANTS, en plus des comptes : c'est ce qui permet de tracer la figure de détection
    ## sans réimplémenter l'appariement ailleurs — deux logiques divergeraient tôt ou tard, et
    ## la figure montrerait alors autre chose que ce que le F2 mesure.
    tp_pairs, fn_idx = [], []
    for g in gt_idx: 
        best = None ## on retient une prédiction pour chaque vraie manoeuvre de gt_idx
        for j,p in enumerate(pred_idx):
            if j in matched:
                continue
            if abs(p-g) <= tolerance : 
                if best is None or abs(p-g) < abs(pred_idx[best] - g):
                    best = j
        if best is None : 
            fn +=1
            fn_idx.append(g)
        else : 
            tp += 1
            matched.add(best)
            distances.append(pred_idx[best] - g)
            tp_pairs.append((g, pred_idx[best]))
    fp_idx = [p for j, p in enumerate(pred_idx) if j not in matched]
    fp = len(pred_idx) - len(matched)
    return {'tp' : tp, 'fp' : fp, 'fn' : fn, 'distances' :distances,
            'tp_pairs' : tp_pairs, 'fn_idx' : fn_idx, 'fp_idx' : fp_idx}

## Calcul des métriques de performance du modèle 
def metrics(tp, fp, fn, distances):
    precision = tp /(tp + fp)  if (tp + fp) > 0 else 0.0
    recall = tp/(tp + fn) if (tp + fn) > 0 else 0.0

    def fbeta(beta, tp, fp,fn):
        beta2 = beta*beta
        denom = (1+beta2)*tp + beta2 * fn + fp 
        return (1+beta2) * tp / denom if denom > 0 else 0.0
    f1 = fbeta(1, tp,fp,fn)
    f2 = fbeta(2, tp,fp,fn)
    d = np.asarray(distances, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(d**2))) if d.size >0 else 0.0

    return {"precision": precision, "recall": recall,
            "f1": f1, "f2": f2, "distances" :d , "rmse": rmse,
            "tp": tp, "fp": fp, "fn": fn}

## On passe 
def evaluate_predictions(gt_events: dict,
                         pred_events : dict,
                         tolerance=matching_tolerance,
                         channels : tuple[str, ...] = ('EW', 'NS')
                         ):
    """Accumule la confusion matrix pour tous les objets sur tous les canaux,
    puis applique les métriques sur la confusion matrix globale"""

    tp = fp= fn = 0
    distances=[]
    for oid in gt_events:
        ## tolerance peut être un dict {oid : tol} : la cadence TLE varie d'un facteur ~4
        ## entre satellites, donc une tolérance en indices ne vaut pas la même durée partout.
        tol = tolerance[oid] if isinstance(tolerance, dict) else tolerance
        for d in channels:
            gt_idx = gt_events[oid][d]
            pred_idx = pred_events[oid][d]
            r =match_events(gt_idx,pred_idx,tol)
            tp += r['tp']
            fp += r['fp']
            fn += r['fn']
            distances += r['distances']
    return metrics(tp, fp, fn, distances)


### EVALUATION DU FINETUNING DORIS
## Même protocole que le localizer, mais les canaux ne sont plus EW/NS : ce sont les deux types
## de manoeuvre, dans l'ordre des colonnes de la cible (L,2).

doris_channels = doris_types


def gt_events_doris(labels,        # DataFrame rendu par load_doris_objects : norad_id, TimeIndex, maneuver_type
                    norad_ids,
                    detection_only=False
    ):
    """{norad : {canal : [TimeIndex..]}}
    Les filtres doivent être EXACTEMENT ceux de build_doris_targets (TimeIndex non NaN,
    maneuver_type connu) : sinon la vérité terrain compte des manoeuvres que la cible n'a
    jamais marquées, et le rappel plafonne sans raison visible.
    detection_only=True fusionne les deux types dans un canal unique 'maneuver'.
    """
    out = {}
    for oid in norad_ids:
        sub = labels[(labels['norad_id'] == oid) & labels['TimeIndex'].notna()
                     & labels['maneuver_type'].isin(doris_types)]
        if detection_only:
            out[oid] = {'maneuver' : sorted(int(x) for x in sub['TimeIndex'].to_numpy())}
            continue
        out[oid] = {maneuver_type : sorted(int(x) for x in
                                           sub[sub['maneuver_type'] == maneuver_type]['TimeIndex'].to_numpy())
                    for maneuver_type in doris_types}
    return out


                        




### FIGURE DE DETECTION
## Une image par objet de validation, pour lire ce que le F2 resume en un chiffre : ou le
## modele tombe juste, ou il invente, ou il rate.

## Serie tracee en fond, par canal. Une manoeuvre cross-track ne laisse pas de trace sur le
## demi-grand-axe -- elle change le plan, pas l'energie de l'orbite -- donc la tracer sur le sma
## montre une courbe plate ou l'evenement est invisible. On lit l'inclinaison a la place.
CANAL_SERIE = {'cross-track': ('inclination', 'inclinaison (deg)'),
               'in-track':    ('sma', 'sma (km)')}
SERIE_DEFAUT = ('sma', 'sma (km)')


def _serie_du_canal(canal, df):
    """(valeurs, libelle) a tracer en fond pour ce canal, avec repli sur le sma.

    Le canal DORIS est le type de manoeuvre ; SPLID utilise EW/NS et 'maneuver' en
    detection_only, aucun des deux ne nommant de type : ils retombent sur le sma.
    """
    col, libelle = CANAL_SERIE.get(str(canal), SERIE_DEFAUT)
    if col not in df.columns:  ## p.ex. le chemin SPLID, qui ne porte pas 'inclination'
        col, libelle = SERIE_DEFAUT
    return df[col].to_numpy(float), libelle


def plot_fold_detection(objects, val_ids, probs_per_obj, gt_events, out_dir,
                        threshold=detection_treshold, channels=('EW', 'NS'),
                        tolerance=matching_tolerance, prefix='detection'):
    """Trace une serie orbitale par canal pour chaque objet de validation, annotee TP / FP / FN.

    La serie depend du canal (cf. CANAL_SERIE) : sma pour l'in-track, inclinaison pour le
    cross-track. Elle est prise BRUTE dans le dataframe de l'objet (km, degres), pas dans les
    features normalisees : une figure de diagnostic doit se lire en unites physiques.

    L'appariement n'est pas refait ici, il vient de match_events -- le meme appel que
    evaluate_predictions -- donc ce que montre la figure est exactement ce que compte le F2,
    au seuil retenu pour le checkpoint trace.

    Une ligne par canal : c'est la granularite a laquelle l'appariement a lieu, et une
    manoeuvre detectee au bon instant mais sur le mauvais canal doit se voir comme un FP
    et un FN, pas comme un succes.

    Renvoie la liste des fichiers ecrits.
    """
    import matplotlib
    matplotlib.use('Agg') ## backend sans affichage : on ecrit des fichiers, pas de fenetre
    import matplotlib.pyplot as plt

    from pathlib import Path
    out_dir = Path(out_dir)
    ecrits = []

    for oid in val_ids:
        seq = probs_per_obj[oid]                      # (L, n_canaux)
        df_obj = objects[oid]
        if len(df_obj) != len(seq):
            raise ValueError(f"norad {oid} : {len(df_obj)} points orbitaux pour {len(seq)} "
                             f"predictions -- series desalignees")
        t = np.arange(len(df_obj))
        tol = tolerance[oid] if isinstance(tolerance, dict) else tolerance
        pred = extract_events(seq, treshold=threshold, channels=channels)

        fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.6 * len(channels) + 1.2),
                                 sharex=True, squeeze=False)
        totaux = {'tp': 0, 'fp': 0, 'fn': 0}

        for ax, canal in zip(axes[:, 0], channels):
            r = match_events(gt_events[oid][canal], pred[canal], tol)
            for cle in totaux:
                totaux[cle] += r[cle]

            serie, libelle = _serie_du_canal(canal, df_obj)
            ax.plot(t, serie, lw=0.7, color='0.35', zorder=1)
            ## TP au niveau de la VERITE terrain, pas de la prediction : la barre marque
            ## l'evenement reel, et le trait horizontal donne l'erreur de datation.
            for g, p in r['tp_pairs']:
                ax.axvline(g, color='tab:green', lw=1.1, alpha=.85, zorder=2)
                ax.plot([g, p], [serie.max(), serie.max()], color='tab:green', lw=1.6, zorder=3)
            for g in r['fn_idx']:
                ax.axvline(g, color='tab:orange', lw=1.1, ls='--', alpha=.9, zorder=2)
            for p in r['fp_idx']:
                ax.axvline(p, color='tab:red', lw=0.9, ls=':', alpha=.75, zorder=2)

            ax.set_ylabel(libelle, fontsize=9)
            ax.set_title(f"{canal} — tp {r['tp']} | fp {r['fp']} | fn {r['fn']}",
                         fontsize=9, loc='left')
            ax.tick_params(labelsize=8)

        axes[-1, 0].set_xlabel('TimeIndex (indice TLE)', fontsize=9)
        poignees = [plt.Line2D([], [], color='tab:green', lw=1.4, label='TP (vraie manoeuvre retrouvee)'),
                    plt.Line2D([], [], color='tab:orange', lw=1.4, ls='--', label='FN (manquee)'),
                    plt.Line2D([], [], color='tab:red', lw=1.2, ls=':', label='FP (inventee)')]
        fig.legend(handles=poignees, loc='lower center', ncol=3, fontsize=8, frameon=False)
        fig.suptitle(f"norad {oid} — seuil {threshold} | tolerance {tol} indices | "
                     f"tp {totaux['tp']} fp {totaux['fp']} fn {totaux['fn']}", fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))

        chemin = out_dir / f"{prefix}_{oid}.png"
        fig.savefig(chemin, dpi=130, bbox_inches='tight')
        plt.close(fig) ## sinon matplotlib garde toutes les figures ouvertes en memoire
        ecrits.append(chemin)

    return ecrits
