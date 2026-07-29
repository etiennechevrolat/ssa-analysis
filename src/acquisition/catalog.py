from spacetrack import like, operators as op
import random
import json
# On fait ici une requete client groupée pour obtenir tous les norad du type de satellites souhaité

def recupIds(client,samples, constellation, orbit_range=None, shuffle=True):
    #Constellation = str récupérée depuis ~/configs/data.yaml
    cstl_name = constellation.name_pattern
    cstl_country= constellation.country
    cstl_norads = constellation.norad_ids

    inf = sup = None
    if orbit_range is not None:
        inf, sup = orbit_range.borneinf, orbit_range.bornesup
    
    print(f"Récupération des IDs {cstl_name or 'ALL_NAMES'}/{cstl_country or 'ALL_COUNTRIES'}.../{cstl_norads or 'ALL_NORADS'}")

    query = dict(
        object_type='PAYLOAD',
        format='json',
        orderby='norad_cat_id',
    )
    
    if inf is not None and sup is not None:
        query['semimajor_axis'] = op.inclusive_range(inf, sup)
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
