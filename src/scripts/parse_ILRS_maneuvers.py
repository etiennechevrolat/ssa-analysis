import pandas as pd
from datetime import datetime, timedelta
from config import load_data_config
from storage.parquet import write_parquet

def doy_to_datetime(year, doy, hour, minute, sec=0.0):
    base=datetime(int(year), 1, 1) + timedelta(
        days=int(doy)-1, 
        hours=int(hour), 
        minutes=int(minute), 
        seconds=float(sec)
    )
    return base

# Les maneuvers sont stockées dans un fichier texte avec les colonnes suivantes :
# id, Year, DOY, Hour, Minute, Year, DOY, Hour, Minute, paramters_type, Year, DOY, hour, minutes, Second, Duration, DeltaV (m/s), accs, delta_acc

def parse_maneuvers(man_path): 
    rows=[]
    with open(man_path) as f:
        for line in f:
            t=line.split()
            if not t :
                continue 
            sat=t[0]
            start_time=doy_to_datetime(t[1], t[2], t[3], t[4])
            end_time=doy_to_datetime(t[5], t[6], t[7], t[8])
            ptype=t[9]
            nburns=int(t[10])
            i = 11 
            for b in range(nburns):
                bparams= t[i:i+15] # 5 dates, 1 durée + 3 delta_v, 3 accs, 3delta_acc
                bepoch = doy_to_datetime(bparams[0], bparams[1], bparams[2], bparams[3], bparams[4])
                bduration= float(bparams[5])
                dv = [float(bparams[6]), float(bparams[7]), float(bparams[8])]
                acc = [float(bparams[9]), float(bparams[10]), float(bparams[11])]
                dacc = [float(bparams[12]), float(bparams[13]), float(bparams[14])]
                i+=15
                rows.append({
                    "sat": sat,
                    "maneuver_start": start_time,
                    "maneuver_end": end_time,
                    "burn_epoch": bepoch,
                    "burn_idx": b,
                    "burn_duration": bduration,
                    "dv_r": dv[0],
                    "dv_t": dv[1],
                    "dv_n": dv[2],
                    "acc_r": acc[0],
                    "acc_t": acc[1],
                    "acc_n": acc[2],
                    "dacc_r": dacc[0],
                    "dacc_t": dacc[1],  
                    "dacc_n": dacc[2]
                })

        return pd.DataFrame(rows)   
    

def main(): 
    config = load_data_config("configs/data.yaml")

    sat_name = config.maneuvers.sat_name
    path_man = f"data/raw/{sat_name}man.txt"
    df = parse_maneuvers(path_man)
    
    tag = f"{sat_name}_man_ILRS"

    write_parquet(df, f"data/raw/{tag}.parquet")
    
if __name__ == "__main__":
    main()