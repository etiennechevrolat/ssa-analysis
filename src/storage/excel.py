def write_excel(df, path):
    df.to_excel(path, index=False)
    print(f"{len(df)} lignes écrites dans {path}")
 