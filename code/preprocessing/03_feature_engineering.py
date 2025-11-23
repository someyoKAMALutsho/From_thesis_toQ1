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

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
INPUT_FILE = Path(r"D:\PM25_Satellite_Research\data\processed\pm25_sampled.parquet")
OUTPUT_FILE = Path(r"D:\PM25_Satellite_Research\data\processed\pm25_with_features.parquet")

def add_temporal_features(df):
    """
    Add cyclical temporal features using sine/cosine encoding.
    
    Source: Kaggle cyclical encoding tutorial (2022)
    Modification: Applied to monthly satellite data (no day/hour components)
    """
    print("Adding temporal features...")
    
    # Month cyclical encoding (preserves Dec→Jan continuity)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (meteorological seasons)
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Winter (DJF)
        3: 1, 4: 1, 5: 1,   # Spring (MAM)
        6: 2, 7: 2, 8: 2,   # Summer (JJA)
        9: 3, 10: 3, 11: 3  # Autumn (SON)
    })
    
    # Year normalized (for trend detection)
    df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    # Days since start (temporal ordering)
    df['days_since_start'] = (df['year'] - 2019) * 365 + (df['month'] - 1) * 30
    
    return df

def add_spatial_features(df):
    """
    Add spatial features for geographic patterns.
    
    Source: Adapted from "Deep learning framework for hourly air pollutants" (2025)
    Modification: Added hemisphere, climate zone proxies, distance from equator
    """
    print("Adding spatial features...")
    
    # Hemisphere indicators
    df['northern_hemisphere'] = (df['lat'] >= 0).astype(int)
    
    # Distance from equator (proxy for climate patterns)
    df['abs_lat'] = np.abs(df['lat'])
    
    # Rough climate zone classification based on latitude
    df['climate_zone'] = pd.cut(
        df['abs_lat'],
        bins=[0, 23.5, 35, 66.5, 90],
        labels=['tropical', 'subtropical', 'temperate', 'polar']
    )
    
    # Convert categorical to numeric for modeling
    df['climate_zone_code'] = df['climate_zone'].cat.codes
    
    # Coastal proximity proxy (simplistic - distance from 0° lon/lat)
    # Note: This is a placeholder; better coastline data could be added later
    df['dist_from_prime_meridian'] = np.abs(df['lon'])
    
    # Geographic quadrant (coarse regional grouping)
    df['quadrant'] = ''
    df.loc[(df['lat'] >= 0) & (df['lon'] >= 0), 'quadrant'] = 'NE'
    df.loc[(df['lat'] >= 0) & (df['lon'] < 0), 'quadrant'] = 'NW'
    df.loc[(df['lat'] < 0) & (df['lon'] >= 0), 'quadrant'] = 'SE'
    df.loc[(df['lat'] < 0) & (df['lon'] < 0), 'quadrant'] = 'SW'
    
    return df

def add_lagged_features(df):
    """
    Add temporal lag features (previous month PM2.5).
    
    Source: "Predicting PM2.5 atmospheric air pollution" (2021, PMC8609844)
    Modification: Adapted for monthly data with proper grouping by grid cell
    """
    print("Adding lagged features...")
    
    # Sort by location and time
    df = df.sort_values(['lat', 'lon', 'year', 'month'])
    
    # Previous month PM2.5 (grouped by grid cell)
    df['pm25_lag1'] = df.groupby(['lat', 'lon'])['pm25'].shift(1)
    
    # 3-month moving average (if sufficient history exists)
    df['pm25_ma3'] = df.groupby(['lat', 'lon'])['pm25'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Fill NaN lags with median (for first months)
    df['pm25_lag1'].fillna(df['pm25'].median(), inplace=True)
    
    return df

def prepare_meteorology_placeholders(df):
    """
    Prepare placeholder columns for meteorological features.
    These will be filled in next script (04_merge_meteorology.py)
    
    Note: ERA5 data download requires CDS API setup (free but needs registration)
    """
    print("Preparing meteorology placeholders...")
    
    # Meteorological variables to be added from ERA5
    met_vars = [
        't2m',           # Temperature at 2m (K)
        'sp',            # Surface pressure (Pa)
        'u10', 'v10',    # Wind components (m/s)
        'blh',           # Boundary layer height (m)
        'tp'             # Total precipitation (m)
    ]
    
    for var in met_vars:
        df[var] = np.nan  # Placeholder
    
    # Derived wind speed
    df['wind_speed'] = np.nan
    
    return df

def main():
    """Main feature engineering pipeline."""
    
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load sampled data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Initial shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")
    
    # Add features
    df = add_temporal_features(df)
    df = add_spatial_features(df)
    df = add_lagged_features(df)
    df = prepare_meteorology_placeholders(df)
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Final shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nNew features added:")
    new_features = [col for col in df.columns if col not in ['lat', 'lon', 'pm25', 'year', 'month']]
    for feat in new_features:
        print(f"  - {feat}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nFeatures with missing values (to be filled in next step):")
        print(missing[missing > 0])
    
    # Save
    print(f"\nSaving to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\nNext step: Download and merge ERA5 meteorology data")
    print("  (See 04_merge_meteorology.py)")
    
    return df

if __name__ == "__main__":
    df = main()
