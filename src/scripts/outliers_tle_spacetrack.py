"""
Dans le but de correctement entrainer les modèles de deep learning, 
on va essayer ici de quantifier précisément les variations de variance sur les tle spacetrack en orbite basse, 
et caractériser les outliers dans les séries temporelles.
"""

import random 
from pathlib import Path 
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust import scale

data_dir  = Path.cwd() / 'data' / 'raw' / 'spacetrack' / 'leo_unlabelled_dataset'

def load_data(data_dir) :
    parquet_files = sorted(data_dir.glob("*.parquet"))
    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(['norad', 'epoch', 'creation_date']).drop_duplicates(['norad', 'epoch'], keep='last').reset_index(drop=True)
    return df
df = load_data(data_dir)

## les outliers_problématiques sont surtout ceux du semi grand axe. 
## on va faire une passe de nettoyage : 
# détection sur une fenetre de TLE des outliers des points qui sont loin de mean +- 3 sigma ET isolés.
def plot_sma(norad, df, start, end):
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

def outliers_detection(norad, df, start, end, n_from_mad = 3, plot = False):
    df_norad = df[df['norad'] == norad].iloc[start:end]
    sma= df_norad['sma'].values
    median = np.median(sma)
    median_abs_deviation = scale.mad(sma)

    is_potential_outlier = np.abs(sma - median) > n_from_mad*median_abs_deviation 

    is_outlier = [False for _ in range(len(is_potential_outlier))]

    for t in range(2, len(is_potential_outlier) -2) :
        if is_potential_outlier[t] :
            if ((is_potential_outlier[t + 1] & is_potential_outlier[t+2]) 
                or (is_potential_outlier[t - 1] & is_potential_outlier[t-2])
                or (is_potential_outlier[t-1] & is_potential_outlier[t+1])
                ): 
                continue   ## le point n'est pas isolé : il fait partie d'une série d'au moins 3 TLE = signifiant 
            else : 
                is_outlier[t] = True
    
    time_index_outliers = np.where(is_outlier)[0]

    if plot :
        t = np.arange(start, end)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.axhline(median)
        ax.axhline(median + n_from_mad * median_abs_deviation)
        ax.axhline(median - n_from_mad * median_abs_deviation)
        ax.scatter(t, sma, marker ='.', color='black')
        ax.scatter(t[is_potential_outlier], sma[is_potential_outlier], marker='.', color='yellow', label='potential outliers')
        ax.scatter(t[is_outlier], sma[is_outlier], marker='.', color='red', label='outliers')
        ax.set(title = f"Sma for norad {norad}", 
            xlabel= "index tle",
            ylabel= "sma")
        ax.legend()
        plt.show()
    return start, time_index_outliers

if __name__ == "__main__" : 
    norads = df['norad'].values

    for norad in random.sample(list(norads), 20) : 
        length_serie = len(df[df['norad'] == norad])
        outliers_detection(norad, df, 0, min(1024,length_serie) , n_from_mad=4, plot=True)
