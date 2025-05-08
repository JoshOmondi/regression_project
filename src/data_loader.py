import pandas as pd

def load_data(path="data/housing.csv"):
    return pd.read_csv(path)
