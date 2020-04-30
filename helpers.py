import pandas as pd
import csv

def csv_to_df(file):
    return pd.read_csv(
        file,
        header=0,
        dtype='str',
        encoding='utf-8',
    )

# def clean_df(df , topic_to_drop):
#     return df[df.topic != topic_to_drop]