
def write_parquet(df, path): 
    df.to_parquet(path, index= False)
    print(f"{len(df)} lignes écrites dans {path}")

