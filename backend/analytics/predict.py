import logging
import os
import pandas as pd
import glob
from prophet import Prophet
from sqlalchemy.orm import Session
import sys

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database.dependencies import SessionLocal
from backend.database.models import Prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In a real system, this would point to a data lake or a well-defined data source.
CRIME_DATA_PARQUET = os.path.join(os.path.dirname(__file__), '../../Data/processed/crime.parquet')
CRIME_DATA_TEST = os.path.join(os.path.dirname(__file__), '../../Data/processed/crime_test.parquet')

# Path to raw UK Police crime data
CRIME_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../Data/UK DATA CRIME 2022 - 2025')

def load_and_aggregate_crime_data(data_dir: str) -> pd.DataFrame:
    """
    Load raw CSV files and aggregate into a national DAILY time series,
    but do the heavy work at NATIONAL MONTHLY level first to avoid
    exploding the dataset.

    Pipeline:
    1. Load all *-street.csv files with Month only (plus other cols if needed).
    2. Convert Month -> month_start (first day of month).
    3. Aggregate to NATIONAL monthly totals.
    4. Evenly distribute each monthly total across its days -> national daily.
    5. Return dataframe {ds: date, y: daily_count}.
    """
    logger.info(f"Loading raw crime data from {data_dir}...")

    all_files = []
    for year_month in os.listdir(data_dir):
        month_dir = os.path.join(data_dir, year_month)
        if os.path.isdir(month_dir):
            csv_files = glob.glob(os.path.join(month_dir, "*-street.csv"))
            all_files.extend(csv_files)

    all_files = sorted(all_files)
    logger.info(f"Found {len(all_files)} CSV files to process")

    dfs = []
    for i, file_path in enumerate(all_files):
        try:
            # We only really need 'Month' to build the time series for Prophet.
            # If you want to keep the other cols for future extensions, that's fine.
            df = pd.read_csv(file_path, usecols=["Month"])
            dfs.append(df)
            if (i + 1) % 10 == 0:
                logger.info(f"Loaded {i + 1}/{len(all_files)} files...")
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue

    logger.info("Combining dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined dataframe shape: {combined_df.shape}")

    # Convert Month to datetime (first day of month)
    combined_df["month_start"] = pd.to_datetime(
        combined_df["Month"] + "-01", format="%Y-%m-%d"
    )

    # 3) NATIONAL monthly totals
    logger.info("Aggregating to NATIONAL monthly totals...")
    monthly_totals = (
        combined_df.groupby("month_start")
        .size()
        .reset_index(name="monthly_total")
        .sort_values("month_start")
    )
    logger.info(
        f"Got {len(monthly_totals)} months from "
        f"{monthly_totals['month_start'].min()} to {monthly_totals['month_start'].max()}"
    )

    # 4) Distribute monthly totals across days (national level only)
    logger.info("Distributing NATIONAL monthly totals to daily...")
    daily_rows = []

    for _, row in monthly_totals.iterrows():
        month_start = row["month_start"]
        total = int(row["monthly_total"])
        days_in_month = pd.Period(month_start, "M").days_in_month

        if total == 0:
            continue

        crimes_per_day = total // days_in_month
        extra_crimes = total % days_in_month

        for day in range(days_in_month):
            date = month_start + pd.Timedelta(days=day)
            daily_count = crimes_per_day + (1 if day < extra_crimes else 0)
            if daily_count > 0:
                daily_rows.append({"ds": date, "y": daily_count})

    daily_df = pd.DataFrame(daily_rows)
    daily_df = daily_df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Created NATIONAL daily time series with {len(daily_df)} records")

    return daily_df

def get_or_build_daily_crime_series() -> pd.DataFrame:
    """
    Get daily crime series, either from cached parquet or build from raw CSVs.
    First run: builds from CSVs and caches to parquet.
    Subsequent runs: instant load from parquet.
    """
    # If cached daily parquet exists, load it
    if os.path.exists(CRIME_DATA_PARQUET):
        logger.info(f"Loading cached daily crime data from {CRIME_DATA_PARQUET}...")
        return pd.read_parquet(CRIME_DATA_PARQUET)

    # Otherwise, build from raw CSVs and cache it
    logger.info("Cached daily crime data not found. Building from raw CSVs...")
    daily_df = load_and_aggregate_crime_data(CRIME_DATA_DIR)

    os.makedirs(os.path.dirname(CRIME_DATA_PARQUET), exist_ok=True)
    daily_df.to_parquet(CRIME_DATA_PARQUET, index=False)
    logger.info(f"Saved daily crime data to {CRIME_DATA_PARQUET}")

    return daily_df

def debug_load_crime_data(max_files=50, expand_to_daily=False):
    """
    Lightweight debug loader for testing:
    - Only reads the first `max_files` CSVs
    - Optionally skips the monthly→daily expansion for faster testing
    """
    import os
    import glob
    from pathlib import Path

    print(f"[DEBUG] Loading up to {max_files} files from {CRIME_DATA_DIR} ...")

    # Find all street crime CSV files
    all_files = []
    for year_month in os.listdir(CRIME_DATA_DIR):
        month_dir = os.path.join(CRIME_DATA_DIR, year_month)
        if os.path.isdir(month_dir):
            csv_files = glob.glob(os.path.join(month_dir, '*-street.csv'))
            all_files.extend(csv_files)

    if not all_files:
        raise RuntimeError(f"No CSV files found in {CRIME_DATA_DIR}")

    # Limit to max_files
    files_to_load = all_files[:max_files]
    print(f"[DEBUG] Will load {len(files_to_load)} out of {len(all_files)} total files")

    # Load and process files
    dfs = []
    for i, file_path in enumerate(files_to_load, start=1):
        try:
            # Load CSV with only needed columns to save memory
            df = pd.read_csv(file_path, usecols=['Month', 'LSOA name', 'Crime type'])
            dfs.append(df)
            if i % 10 == 0 or i == len(files_to_load):
                print(f"[DEBUG] Loaded {i}/{len(files_to_load)} files...")
        except Exception as e:
            print(f"[DEBUG] Warning: Error loading {file_path}: {e}")
            continue

    # Combine dataframes
    df = pd.concat(dfs, ignore_index=True)
    print(f"[DEBUG] Combined dataframe shape (raw): {df.shape}")

    if expand_to_daily:
        print("[DEBUG] Expanding monthly to daily (this might be heavy)...")
        # Convert Month to datetime (first day of month)
        df['month_start'] = pd.to_datetime(df['Month'] + '-01', format='%Y-%m-%d')

        # Distribute monthly crimes across days in the month
        daily_rows = []

        # For debug, limit to first few months to avoid huge expansion
        unique_months = df['month_start'].unique()[:3]  # Only first 3 months
        print(f"[DEBUG] Expanding only {len(unique_months)} months for testing...")

        for month_start in unique_months:
            month_data = df[df['month_start'] == month_start]
            month_crimes = len(month_data)
            days_in_month = pd.Period(month_start, 'M').days_in_month

            # Distribute crimes evenly across days
            crimes_per_day = month_crimes // days_in_month
            extra_crimes = month_crimes % days_in_month

            for day in range(days_in_month):
                date = month_start + pd.Timedelta(days=day)
                daily_count = crimes_per_day + (1 if day < extra_crimes else 0)

                if daily_count > 0:
                    daily_rows.append({
                        'ds': date,
                        'y': daily_count
                    })

        # Create final daily dataframe
        df = pd.DataFrame(daily_rows)
        print(f"[DEBUG] After daily expansion: {len(df)} records")

    print(f"[DEBUG] Final dataframe shape: {df.shape}")
    return df

def run_prediction_pipeline(db: Session, df: pd.DataFrame):
    """
    Runs the prediction pipeline using daily aggregated time series.
    Trains Prophet on daily counts and forecasts 30-365 days ahead.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping prediction.")
        return

    # Clear old predictions
    logger.info("Clearing old predictions from the database...")
    db.query(Prediction).delete()
    db.commit()

    # Aggregate to national daily totals for forecasting
    logger.info("Aggregating to national daily totals...")
    national_daily = df.groupby('ds')['y'].sum().reset_index()
    national_daily = national_daily.sort_values('ds')

    logger.info(f"National daily series: {len(national_daily)} days from {national_daily['ds'].min()} to {national_daily['ds'].max()}")

    # Train Prophet model on national daily totals
    logger.info("Training Prophet model on national daily crime counts...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # More flexible trend changes
        seasonality_prior_scale=10.0   # Strong seasonal patterns
    )

    model.fit(national_daily)

    # Forecast 30-365 days ahead
    logger.info("Generating forecasts for 30-365 days ahead...")
    future_periods = 365
    future = model.make_future_dataframe(periods=future_periods, freq='D')
    forecast = model.predict(future)

    # Save predictions to database (national level, no area/crime type breakdown)
    logger.info("Saving forecast results to database...")
    predictions_to_save = []

    # Only save the future predictions (30-365 days ahead)
    forecast_future = forecast[forecast['ds'] > national_daily['ds'].max()].copy()

    for _, row in forecast_future.iterrows():
        pred = Prediction(
            area_id='NATIONAL',  # National level predictions
            ts=row['ds'],
            crime_type='ALL_CRIMES',  # All crime types aggregated
            yhat=max(0, row['yhat']),  # Ensure non-negative
            yhat_lower=max(0, row['yhat_lower']),
            yhat_upper=max(0, row['yhat_upper'])
        )
        predictions_to_save.append(pred)

    db.bulk_save_objects(predictions_to_save)
    db.commit()

    logger.info(f"Saved {len(predictions_to_save)} daily forecast predictions (30-365 days ahead)")

    # Log forecast summary
    future_30 = forecast_future.head(30)
    future_365 = forecast_future.tail(1)

    logger.info("Forecast Summary:")
    logger.info(f"  30 days ahead: {future_30['yhat'].mean():.1f} crimes/day (range: {future_30['yhat_lower'].min():.1f} - {future_30['yhat_upper'].max():.1f})")
    logger.info(f"  365 days ahead: {future_365['yhat'].iloc[0]:.1f} crimes/day (range: {future_365['yhat_lower'].iloc[0]:.1f} - {future_365['yhat_upper'].iloc[0]:.1f})")


if __name__ == "__main__":
    logger.info("Starting crime prediction pipeline...")
    db = SessionLocal()
    try:
        crime_df = get_or_build_daily_crime_series()
        if crime_df.empty:
            logger.error("Failed to load crime data. Exiting.")
            sys.exit(1)

        run_prediction_pipeline(db, crime_df)
    finally:
        db.close()
