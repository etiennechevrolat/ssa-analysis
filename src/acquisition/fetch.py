import json
from collections import defaultdict


# Ici on va a partir d'une liste d'IDs récupérer par batch les historiques des satellites
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i: i+size]

def fetch_history_batched(client, sat_ids, period, batch_size=50):
    all_records = []
    for batch in chunked(sat_ids,batch_size):
        raw = client.gp_history(
            norad_cat_id=batch,
            epoch = f">now-{period}",
            orderby='norad_cat_id,epoch',
            format= 'json'
        )
        records = json.loads(raw)
        all_records.extend(records)
    return all_records

def group_by_sat(records): 
    grouped = defaultdict(list)
    for e in records:
        grouped[int(e['NORAD_CAT_ID'])].append(e)
    return grouped
