from spacetrack import like 
import random
import json

# On fait ici une requete client groupée pour obtenir tous les norad du type de satellites souhaité


def recupIds(client,samples, constellation, shuffle=True):
    #Constellation = str récupérée depuis ~/configs/data.yaml
    print(f"Récupération des IDs {constellation}...")
    raw = client.gp(
        object_name=like(f'{constellation}%'),
        object_type='PAYLOAD',
        format='json',
        orderby='norad_cat_id',
    )
    satellites = json.loads(raw)
    print(f"{len(satellites)} satellites trouvés.")

    sat_ids = [e['NORAD_CAT_ID'] for e in satellites]
    
    if shuffle:
        sat_ids_copy = sat_ids.copy()
        random.shuffle(sat_ids_copy)
        return sat_ids_copy[:samples]
    else:
        return sat_ids[:samples]
