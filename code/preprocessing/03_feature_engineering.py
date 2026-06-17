"""
Feature Engineering for Satellite PM2.5 Data
Adds temporal, spatial, and prepares for meteorological features.

ORIGINAL SOURCE INSPIRATION:
- Cyclical encoding: Kaggle "Encoding Cyclical Features for Deep Learning" (2022)
  https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning
- PM2.5 feature engineering: "Deep learning framework for hourly air pollutants" 
  (2025, PMC12216515) https://pmc.ncbi.nlm.nih.gov/articles/PMC12216515/
- Temporal features: "Predicting PM2.5 atmospheric air pollution using deep learning" 
  (2021, PMC8609844) https://pmc.ncbi.nlm.nih.gov/articles/PMC8609844/

MODIFICATIONS FOR THIS PROJECT:
- Adapted for Washington University SAT PM2.5 V6GL02.04 format
- Added spatial grid features (distance calculations, quadrants)
- Structured for D: drive file paths
- Prepared for ERA5 meteorology integration (placeholder structure)
- Optimized for memory efficiency with 1M+ samples
- Added feature importance analysis preparation for LightGBM
"""

"""
Feature Engineering for Satellite PM2.5 Data
Adds temporal, spatial, and lagged features.

MODIFIED (v2):
- Removed 'wind_speed' placeholder (no longer used; derived in MERRA-2
  integration previously, now excluded to avoid duplication and
  multicollinearity issues).
- Kept script focused on PM2.5 + temporal/spatial structure;
  meteorology is now assumed to be merged later from reanalysis.
- Updated paths to F: drive.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
INPUT_FILE  = PROJECT_DIR / "data" / "processed" / "pm25_sampled.parquet"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "pm25_with_features.parquet"

def add_temporal_features(df):
    """
    Add cyclical temporal features using sine/cosine encoding.
    """
    print("Adding temporal features...")

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,
         3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2,
         9: 3,10: 3,11: 3
    })

    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df['days_since_start'] = (df['year'] - 2019) * 365 + (df['month'] - 1) * 30

    return df

def add_spatial_features(df):
    """
    Add spatial features for geographic patterns.
    """
    print("Adding spatial features...")

    df['northern_hemisphere'] = (df['lat'] >= 0).astype(int)
    df['abs_lat'] = np.abs(df['lat'])

    df['climate_zone'] = pd.cut(
        df['abs_lat'],
        bins=[0, 23.5, 35, 66.5, 90],
        labels=['tropical', 'subtropical', 'temperate', 'polar']
    )
    df['climate_zone_code'] = df['climate_zone'].cat.codes

    df['dist_from_prime_meridian'] = np.abs(df['lon'])

    df['quadrant'] = ''
    df.loc[(df['lat'] >= 0) & (df['lon'] >= 0), 'quadrant'] = 'NE'
    df.loc[(df['lat'] >= 0) & (df['lon'] <  0), 'quadrant'] = 'NW'
    df.loc[(df['lat'] <  0) & (df['lon'] >= 0), 'quadrant'] = 'SE'
    df.loc[(df['lat'] <  0) & (df['lon'] <  0), 'quadrant'] = 'SW'

    return df

def add_lagged_features(df):
    """
    Add temporal lag features (previous month PM2.5 and 3‑month moving average).
    """
    print("Adding lagged features...")

    df = df.sort_values(['lat', 'lon', 'year', 'month'])

    df['pm25_lag1'] = df.groupby(['lat', 'lon'])['pm25'].shift(1)
    df['pm25_ma3'] = df.groupby(['lat', 'lon'])['pm25'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    df['pm25_lag1'].fillna(df['pm25'].median(), inplace=True)

    return df

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Initial shape   : {df.shape}")
    print(f"Initial columns : {list(df.columns)}")

    df = add_temporal_features(df)
    df = add_spatial_features(df)
    df = add_lagged_features(df)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Final shape   : {df.shape}")
    print(f"Total columns : {len(df.columns)}")

    new_features = [c for c in df.columns if c not in ['lat', 'lon', 'pm25', 'year', 'month']]
    print("\nNew features added:")
    for feat in new_features:
        print(f"  - {feat}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nColumns with missing values (to be handled later):")
        print(missing[missing > 0])

    print(f"\nSaving to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\n✓ Feature engineering file saved.")
    return df

if __name__ == "__main__":
    df = main()
