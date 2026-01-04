import polars as pl
from pathlib import Path

path = Path(r"C:\Users\Paras\Desktop\Police App\police-3d-face-app\Data\processed\crime.parquet")
print("Exists:", path.exists(), "Size:", path.stat().st_size if path.exists() else None)

df = pl.read_parquet(path)
print("Shape:", df.shape)
print("Columns:", df.columns)

# Show inferred dtypes
print(df.dtypes)

# Peek at Month raw values (first 15)
if "Month" in df.columns:
    print("First 15 Month values:", df.select("Month").head(15))

# Null counts for critical columns
critical = [c for c in ["Month", "Falls within", "LSOA name", "Crime type"] if c in df.columns]
print("Null counts:", df.select([pl.col(c).is_null().sum().alias(c) for c in critical]))

# Distinct counts
if critical:
    print("Distinct counts:", df.select([pl.col(c).n_unique().alias(c+"_nuniq") for c in critical]))

# Show a small sample of relevant columns
show_cols = [c for c in ["Month","Falls within","LSOA name","Crime type"] if c in df.columns]
print(df.select(show_cols).head(10))