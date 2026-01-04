import pandas as pd
import sys
sys.path.append('backend')
from backend.analytics.predict import load_and_prepare_data, CRIME_DATA_PARQUET

print('Loading crime.parquet...')
df = load_and_prepare_data(CRIME_DATA_PARQUET)
print(f'Loaded {len(df)} crime records')
print(f'Unique areas: {len(df["LSOA name"].unique())}')
print(f'Unique crime types: {len(df["Crime type"].unique())}')
print(f'Date range: {df["ds"].min()} to {df["ds"].max()}')
print()
print('Sample data:')
print(df.head())
print()
print('Top areas by crime count:')
area_counts = df.groupby('LSOA name').size().sort_values(ascending=False).head(10)
for area, count in area_counts.items():
    print(f'{area}: {count} crimes')