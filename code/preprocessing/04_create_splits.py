"""
Create temporal and spatial train/test splits for rigorous validation.

ORIGINAL SOURCE INSPIRATION:
- LOCO validation: "National PM2.5 and NO2 exposure models for China" (2018)
  https://depts.washington.edu/airqual/Marshall_100.pdf
- Temporal split strategy: "Deep learning framework for hourly air pollutants" (2025)
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12216515/

MODIFICATIONS FOR THIS PROJECT:
- Adapted for monthly satellite data (2019-2023)
- Chronological split at 2022-07 (80/20 temporal split)
- Added regional LOCO folds based on geographic quadrants
- Optimized for 1M+ samples on D: drive
- Prepared for GPU-accelerated model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
INPUT_FILE = Path(r"D:\PM25_Satellite_Research\data\processed\pm25_with_features.parquet")
OUTPUT_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")

# Temporal split date (train up to June 2022, test from July 2022 onwards)
SPLIT_YEAR = 2022
SPLIT_MONTH = 7

def create_temporal_split(df):
    """
    Chronological train/test split.
    Critical for time series: prevents data leakage.
    """
    print("Creating temporal split...")
    
    # Create temporal ordering
    df['date_key'] = df['year'] * 100 + df['month']
    split_key = SPLIT_YEAR * 100 + SPLIT_MONTH
    
    # Split
    train_temporal = df[df['date_key'] < split_key].copy()
    test_temporal = df[df['date_key'] >= split_key].copy()
    
    # Drop helper column
    train_temporal.drop('date_key', axis=1, inplace=True)
    test_temporal.drop('date_key', axis=1, inplace=True)
    
    print(f"  Train: {len(train_temporal):,} samples ({train_temporal['year'].min()}-{train_temporal['month'].min():02d} to {train_temporal['year'].max()}-{train_temporal['month'].max():02d})")
    print(f"  Test:  {len(test_temporal):,} samples ({test_temporal['year'].min()}-{test_temporal['month'].min():02d} to {test_temporal['year'].max()}-{test_temporal['month'].max():02d})")
    
    return train_temporal, test_temporal

def create_spatial_loco_folds(df):
    """
    Leave-One-Region-Out (LOCO) validation using geographic quadrants.
    Tests spatial generalization ability.
    """
    print("\nCreating spatial LOCO folds...")
    
    # Use quadrants for LOCO (NE, NW, SE, SW)
    quadrants = df['quadrant'].unique()
    
    loco_folds = {}
    for quad in quadrants:
        test_loco = df[df['quadrant'] == quad].copy()
        train_loco = df[df['quadrant'] != quad].copy()
        
        loco_folds[quad] = {
            'train': train_loco,
            'test': test_loco
        }
        
        print(f"  {quad}: Train={len(train_loco):,}, Test={len(test_loco):,}")
    
    return loco_folds

def main():
    """Main splitting pipeline."""
    
    print("="*60)
    print("TRAIN/TEST SPLIT CREATION")
    print("="*60)
    
    # Load feature-engineered data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Total samples: {len(df):,}")
    
    # Remove rows with NaN meteorology (we don't have it yet)
    # Keep only non-meteorology features for now
    non_met_cols = [col for col in df.columns if col not in ['t2m', 'sp', 'u10', 'v10', 'blh', 'tp', 'wind_speed']]
    df_clean = df[non_met_cols].copy()
    
    print(f"Using {len(df_clean.columns)} features (excluding meteorology placeholders)")
    
    # 1. Temporal split
    train_temporal, test_temporal = create_temporal_split(df_clean)
    
    # 2. Spatial LOCO folds
    loco_folds = create_spatial_loco_folds(df_clean)
    
    # Save splits
    print("\n" + "="*60)
    print("SAVING SPLITS")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Temporal split
    train_temporal.to_parquet(OUTPUT_DIR / "train_temporal.parquet", index=False)
    test_temporal.to_parquet(OUTPUT_DIR / "test_temporal.parquet", index=False)
    print(f"✓ Saved temporal splits")
    
    # LOCO folds
    loco_dir = OUTPUT_DIR / "loco_folds"
    loco_dir.mkdir(exist_ok=True)
    
    for quad, fold_data in loco_folds.items():
        fold_data['train'].to_parquet(loco_dir / f"train_loco_{quad}.parquet", index=False)
        fold_data['test'].to_parquet(loco_dir / f"test_loco_{quad}.parquet", index=False)
    
    print(f"✓ Saved {len(loco_folds)} LOCO folds to: {loco_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print(f"Temporal Train: {len(train_temporal):,} samples")
    print(f"Temporal Test:  {len(test_temporal):,} samples")
    print(f"LOCO folds: {len(loco_folds)} geographic regions")
    
    print("\nReady for model training!")
    print("  Next: Run 05_train_baseline_models.py")

if __name__ == "__main__":
    main()
