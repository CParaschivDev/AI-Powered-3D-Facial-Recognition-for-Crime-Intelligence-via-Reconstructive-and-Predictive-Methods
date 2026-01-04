import pandas as pd

print('Analyzing crime data...')
df = pd.read_parquet('Data/processed/crime.parquet')

print(f'Total rows: {len(df)}')
print()

print('Missing values:')
print(df.isnull().sum())
print()

print('Crime type distribution:')
crime_types = df['Crime type'].value_counts()
print(crime_types.head(10))
print()

print('LSOA name missing:')
missing_lsoa = df['LSOA name'].isnull().sum()
valid_lsoa = df['LSOA name'].notnull().sum()
print(f'Missing LSOA names: {missing_lsoa}')
print(f'Valid LSOA names: {valid_lsoa}')
print()

print('Month format:')
print(df['Month'].head())
print()

# Check if Month can be parsed
try:
    df['ds'] = pd.to_datetime(df['Month'], format='%Y-%m')
    print('Month parsing successful')
    print(f'Date range: {df["ds"].min()} to {df["ds"].max()}')
except Exception as e:
    print(f'Month parsing failed: {e}')

# Simulate the load_and_prepare_data logic
print()
print('Simulating load_and_prepare_data...')
df_clean = df.copy()
df_clean['ds'] = pd.to_datetime(df_clean['Month'], format='%Y-%m')
df_clean['y'] = 1

# Filter out rows with missing LSOA name or Crime type
before_filter = len(df_clean)
df_clean = df_clean[df_clean['LSOA name'].notnull() & (df_clean['Crime type'] != 'None')]
after_filter = len(df_clean)

print(f'Rows before filtering: {before_filter}')
print(f'Rows after filtering: {after_filter}')
print(f'Filtered out: {before_filter - after_filter} rows')

if after_filter > 0:
    df_agg = df_clean.groupby(['ds', 'LSOA name', 'Crime type']).count().reset_index()
    df_agg = df_agg[['ds', 'LSOA name', 'Crime type', 'y']]
    print(f'Aggregated to {len(df_agg)} records')
    print('Sample aggregated data:')
    print(df_agg.head())
else:
    print('No valid data after filtering!')