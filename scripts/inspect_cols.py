import pandas as pd
import sys

try:
    df = pd.read_csv('main.csv', nrows=1)
    cols = df.columns.tolist()
    for i, col in enumerate(cols):
        print(f"{i}: {col}")
except Exception as e:
    print(e)
