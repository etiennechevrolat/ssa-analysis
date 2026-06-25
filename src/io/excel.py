from ssa.orbital import GM, GM13, MRAD, PI, TPI86, PERIOD
from src.acquisition.fetch import fetch_history_batched, group_by_sat

def writing_data_in_excel(sat_ids, client, workbook, worksheet, formats, wsline):
    
    all_records = fetch_history_batched(client, sat_ids)
    grouped_by_sat= group_by_sat(all_records)

    for satid in grouped_by_sat:
        print(f"Scanning satellite {satid}")
        try:
            history = grouped_by_sat[satid]
            if not history : 
                print(f"Réponse vide pour {satid}, satellite ignoré.")
            else:
                print(f"Satellite {satid} : {len(history)} relevés sur {PERIOD}j.")
                for e in history:
                    mmoti = float(e['MEAN_MOTION'])
                    ecc   = float(e['ECCENTRICITY'])
                    sma   = GM13 / ((TPI86 * mmoti) ** (2.0 / 3.0)) / 1000.0
                    apo   = sma * (1.0 + ecc) - MRAD
                    per   = sma * (1.0 - ecc) - MRAD
                    smak  = sma * 1000.0
                    orbT  = 2.0 * PI * ((smak ** 3.0) / GM) ** 0.5
                    orbV  = (GM / smak) ** 0.5

                    worksheet.write(wsline, 0,  int(e['NORAD_CAT_ID']))
                    worksheet.write(wsline, 1,  e['OBJECT_NAME'])
                    worksheet.write(wsline, 2,  e['EPOCH'])
                    worksheet.write(wsline, 3,  float(e['REV_AT_EPOCH']))
                    worksheet.write(wsline, 4,  float(e['INCLINATION']),       formats['z1'])
                    worksheet.write(wsline, 5,  ecc,                           formats['z3'])
                    worksheet.write(wsline, 6,  mmoti,                         formats['z1'])
                    worksheet.write(wsline, 7,  apo,                           formats['z1'])
                    worksheet.write(wsline, 8,  per,                           formats['z1'])
                    worksheet.write(wsline, 9,  (apo + per) / 2.0,            formats['z1'])
                    worksheet.write(wsline, 10, float(e['RA_OF_ASC_NODE']),    formats['z1'])
                    worksheet.write(wsline, 11, float(e['ARG_OF_PERICENTER']), formats['z1'])
                    worksheet.write(wsline, 12, float(e['MEAN_ANOMALY']),      formats['z1'])
                    worksheet.write(wsline, 13, sma,                           formats['z1'])
                    worksheet.write(wsline, 14, orbT,                          formats['z0'])
                    worksheet.write(wsline, 15, orbV,                          formats['z0'])
                    wsline += 1

                    if wsline > 1048570:
                        print("Limite Excel atteinte, arrêt.")
                        break

        except Exception as err:
            print(f"Erreur satellite {satid} : {err}")

    return wsline