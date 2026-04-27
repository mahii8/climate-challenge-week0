import pandas as pd
import numpy as np
import os

def load_all_data(data_path='data'):
    countries = ['ethiopia', 'kenya', 'sudan', 'tanzania', 'nigeria']
    dfs = []
    for country in countries:
        filepath = os.path.join(data_path, f'{country}_clean.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Country'] = country.capitalize()
            df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def filter_data(df, countries, year_range, variable):
    filtered = df[
        (df['Country'].isin(countries)) &
        (df['Date'].dt.year >= year_range[0]) &
        (df['Date'].dt.year <= year_range[1])
    ]
    return filtered