import pandas as pd
from acquisition.client import initClient
from acquisition.catalog import recupIds
from ssa.orbital import derive
from acquisition.fetch import fetch_history_batched, group_by_sat
from storage.excel import write_excel
from storage.parquet import write_parquet
from config import load_data_config

def build_dataframe(grouped): 
    #grouped = dico des historiques des satellites, après être passé par fetch 
    rows = []
    for satids, history in grouped.items():
        for e in history : 
            MeanMotion, ecc = e['MEAN_MOTION'], e['ECCENTRICITY']
            rows.append(
                {
                    'norad' : int(e['NORAD_CAT_ID']),
                    'object_name' : e['OBJECT_NAME'],
                    'epoch' : e['EPOCH'], 
                    'rev_at_epoch' : e['REV_AT_EPOCH'],
                    'inclination' : e['INCLINATION'],
                    "raan": float(e["RA_OF_ASC_NODE"]),
                    "arg_perigee":   float(e["ARG_OF_PERICENTER"]),
                    "mean_anomaly":  float(e["MEAN_ANOMALY"]),
                    "mean_motion":   MeanMotion,
                    "eccentricity":  ecc,
                    **derive(float(MeanMotion), float(ecc)),
                }
            )
    return pd.DataFrame(rows)
 
def main(): 

    #Client SpaceTrack
    client =initClient("SpaceTrack.ini")
    config = load_data_config("configs/data.yaml")
    for constellation in config.constellations: 
        #Parametrès de la requete : samples = nombre d'ids différents, period = durée de l'historique
        samples = 5
        period = 30

        #Recup les ids
        satids = recupIds(client, samples, constellation )

        #Fetching, on récupère un dico avec les données rangées par ids, epochs
        records = fetch_history_batched(client, satids, period)
        grouped = group_by_sat(records)

        df = build_dataframe(grouped)
        
        #On écrit dans data
        tag = f"{constellation.name_pattern or 'ALL'}_{constellation.country or 'ALL'}"
        write_excel(df, f"data/raw/{tag}.xlsx")
        

if __name__=="__main__": 
    main()
