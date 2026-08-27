"""
Dans le but de correctement entrainer les modèles de deep learning, 
on va essayer ici de quantifier précisément les variations de variance sur les tle spacetrack en orbite basse, 
et caractériser les outliers dans les séries temporelles.
"""


from pathlib import Path 
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data_dir  = Path.cwd() / 'data' / 'raw' / 'spacetrack' / 'leo_unlabelled_dataset'

def load_data(data_dir) :
    df = pd.concat([pd.read_parquet(p) for p in data_dir.glob("*.parquet")], ignore_index=True)
    return df
df = load_data(data_dir)

def plot_sma_local(norad, df, start, end):
    """
    Start/end sont des entiers correspondant au numéro des TLE depuis le premier TLE du dataframe de ce norad
    """
    if not norad in df['norad'] : 
        print("pas trouvé")
    else : 
        df_norad = df[df['norad'] == norad]
        subset = df_norad.iloc[start:end]
        t = np.arange(start, end)
        x = subset['sma'].values

        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(t, x, marker ='.')
        ax.set(title = f"Sma for norad {norad}", 
            xlabel= "index tle",
            ylabel= "sma")
        plt.show()

if __name__ == "__main__" : 
    plot_sma_local(49217, df, 0, 1024)
    