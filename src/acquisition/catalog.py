from spacetrack import like, operators as op
import random
import json
# On fait ici une requete client groupée pour obtenir tous les norad du type de satellites souhaité

def recupIds(client,samples, constellation, orbit_range=None, epoch_min=None, shuffle=True):
    #Constellation = str récupérée depuis ~/configs/data.yaml
    cstl_name = constellation.name_pattern
    cstl_country= constellation.country
    cstl_norads = constellation.norad_ids

    inf = sup = None
    if orbit_range is not None:
        inf, sup = orbit_range.borneinf, orbit_range.bornesup
    
    print(f"Récupération des IDs {cstl_name or 'ALL_NAMES'}/{cstl_country or 'ALL_COUNTRIES'}.../{cstl_norads or 'ALL_NORADS'}"
          f"{f' avec element set posterieur a {epoch_min}' if epoch_min else ''}")

    query = dict(
        object_type=['PAYLOAD','DEBRIS'],
        format='json',
        orderby='norad_cat_id',
    )
    
    if inf is not None and sup is not None:
        query['semimajor_axis'] = op.inclusive_range(inf, sup)
    if epoch_min is not None:
        ## La classe gp ne contient qu'un element set par objet : le dernier connu. Sans ce filtre
        ## on selectionne aussi les objets rentres il y a des annees, dont le dernier etat satisfait
        ## encore le critere de demi-grand axe, mais dont gp_history ne renverra rien sur la fenetre
        ## demandee. Mesure sur la requete LEO : 55 663 objets sans le filtre, 30 420 avec, et le
        ## rendement passe de 9 a 50 objets utiles par batch de 50.
        ## On filtre sur epoch et NON sur decay_date : un objet rentre PENDANT la fenetre garde un
        ## historique exploitable, c'est meme la que la signature de trainee atmospherique est nette.
        query['epoch'] = op.greater_than(epoch_min)
    if cstl_name:
        query['object_name'] = like(f"{cstl_name}%")
    if cstl_country:  
        query['country_code'] = cstl_country
    if cstl_norads:
        query['norad_cat_id'] = cstl_norads

    raw = client.gp(**query)
    satellites = json.loads(raw)
    print(f"{len(satellites)} satellites trouvés.")

    sat_ids = [e['NORAD_CAT_ID'] for e in satellites]
    
    if shuffle:
        sat_ids_copy = sat_ids.copy()
        random.shuffle(sat_ids_copy)
        return sat_ids_copy[:samples]
    else:
        return sat_ids[:samples]
