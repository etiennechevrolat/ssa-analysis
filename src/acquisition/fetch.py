import json
import time
import httpx
from collections import defaultdict
from spacetrack import operators as op

# Ici on va a partir d'une liste d'IDs récupérer par batch les historiques des satellites
def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i: i+size]


def _fetch_batch(client, batch, start, end, retries=3):
    """Récupère l'historique d'un batch d'IDs avec retry/backoff sur timeout.
    En cas d'échecs répétés, scinde le batch en deux et réessaie récursivement.
    """
    for attempt in range(retries):
        try:
            raw = client.gp_history(
                norad_cat_id=batch,
                epoch=op.inclusive_range(start, end),
                orderby='norad_cat_id,epoch',
                format='json',
            )
            return json.loads(raw)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            wait = 2 ** attempt
            print(f"  Timeout sur batch de {len(batch)} sats (essai {attempt+1}/{retries}), "
                  f"nouvelle tentative dans {wait}s...")
            time.sleep(wait)

    # Échecs répétés : on scinde le batch si possible
    if len(batch) > 1:
        mid = len(batch) // 2
        print(f"  Scission du batch ({len(batch)} → {mid}+{len(batch)-mid})")
        return (_fetch_batch(client, batch[:mid], start, end, retries)
                + _fetch_batch(client, batch[mid:], start, end, retries))

    print(f"  ÉCHEC définitif pour le satellite {batch}, ignoré.")
    return []


def fetch_history_batched(client, sat_ids, start, end, batch_size=50):
    all_records = []
    batches = list(chunked(sat_ids, batch_size))
    for n, batch in enumerate(batches, 1):
        print(f"Batch {n}/{len(batches)} ({len(batch)} satellites)...")
        all_records.extend(_fetch_batch(client, batch, start, end))
    return all_records

def group_by_sat(records): 
    grouped = defaultdict(list)
    for e in records:
        grouped[int(e['NORAD_CAT_ID'])].append(e)
    return grouped
