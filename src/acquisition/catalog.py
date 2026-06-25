from spacetrack import like 
import random
import json

# On fait ici une requete client groupée pour obtenir tous les norad du type de satellites souhaité

def recupIds(client,samples, constellation, shuffle=True):
    #Constellation = str récupérée depuis ~/configs/data.yaml
    cstl_name, country= constellation.name_pattern, constellation.country 
    print(f"Récupération des IDs {cstl_name or 'ALL'}/{country or 'ALL'}...")

    query = dict(
        object_type='PAYLOAD',
        format='json',
        orderby='norad_cat_id',
    )
    if cstl_name:
        query['object_name'] = cstl_name
    if country:  
        query['country_code'] = country
    if not cstl_name and not country:
        query['limit'] = max(samples * 5, 500)   # limite la taille de requete si pas de précision de pays/constellation
    
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
