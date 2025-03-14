import polars as pl


def load_and_process_data():
    df_skills = pl.read_csv('data/Adecco_csv_parsed_data/parsed_data.csv', separator=';').rename({'CANDIDATE_ID':'cand_id'})
    df_info = pl.read_csv('data/Adecco_dataset/reverse_matching_20240213.csv', separator=';')
    df_info = df_info.with_columns(df_info.select(pl.col("cand_id").str.replace_all(',','',literal=True).cast(pl.Int64).alias('cand_id')))
    df_total = df_skills.join(df_info, on='cand_id')
    
    return df_skills, df_info, df_total





