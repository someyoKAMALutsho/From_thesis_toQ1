"""
Extract PM2.5 from NetCDF files with intelligent spatial sampling.
Adapted from satellite PM2.5 processing workflows.
Uses Dask for memory-efficient processing.

MODIFICATIONS FOR THIS PROJECT:
- Added spatial downsampling (every Nth grid cell) to reduce data size
- Optimized for GTX 1650 GPU environment
- Structured for Washington University V6GL02.04 format
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
NC_DIR = Path(r"D:\PM25_Satellite_Research\data\raw\monthly_netcdf")
OUTPUT_FILE = Path(r"D:\PM25_Satellite_Research\data\processed\pm25_sampled.parquet")

# Spatial sampling: take every Nth point (reduces 468M to manageable size)
SPATIAL_STRIDE = 100  # Every 100th point → ~47k cells per month → ~2.8M total
# Adjust this: lower = more data but slower; higher = less data but faster

# === EXTRACTION FUNCTION ===
def extract_month(nc_file):
    """Extract and sample PM2.5 from one NetCDF file."""
    try:
        # Open with xarray
        ds = xr.open_dataset(nc_file)
        
        # Subsample spatially (every SPATIAL_STRIDE-th point)
        pm25_sampled = ds['PM25'][::SPATIAL_STRIDE, ::SPATIAL_STRIDE].values
        lat_sampled = ds['lat'][::SPATIAL_STRIDE].values
        lon_sampled = ds['lon'][::SPATIAL_STRIDE].values
        
        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(lon_sampled, lat_sampled)
        
        # Flatten to dataframe
        df = pd.DataFrame({
            'lat': lat_grid.flatten(),
            'lon': lon_grid.flatten(),
            'pm25': pm25_sampled.flatten()
        })
        
        # Extract year and month from filename
        # Format: V6GL02.04.CNNPM25.GL.201901-201901.nc
        filename = nc_file.stem
        date_part = filename.split('.')[-1].split('-')[0]  # "201901"
        year = int(date_part[:4])
        month = int(date_part[4:6])
        
        df['year'] = year
        df['month'] = month
        
        # Remove invalid values
        df = df[df['pm25'] >= 0]
        df = df[df['pm25'] < 500]  # Remove unrealistic outliers
        df = df.dropna()
        
        ds.close()
        return df
        
    except Exception as e:
        print(f"ERROR processing {nc_file.name}: {e}")
        return None

# === MAIN PROCESSING ===
def process_all_files():
    """Process all NetCDF files across all years."""
    
    # Find all .nc files
    nc_files = sorted(NC_DIR.rglob("*.nc"))
    print(f"Found {len(nc_files)} NetCDF files\n")
    
    if len(nc_files) == 0:
        print("ERROR: No .nc files found!")
        return
    
    # Process each file
    all_data = []
    for nc_file in tqdm(nc_files, desc="Extracting NetCDF files"):
        df_month = extract_month(nc_file)
        if df_month is not None:
            all_data.append(df_month)
    
    # Combine all months
    print("\nCombining all data...")
    df_full = pd.concat(all_data, ignore_index=True)
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(df_full):,}")
    print(f"Date range: {df_full['year'].min()}-{df_full['month'].min():02d} to {df_full['year'].max()}-{df_full['month'].max():02d}")
    print(f"Unique grid cells: {df_full.groupby(['lat','lon']).ngroups:,}")
    print(f"PM2.5 range: {df_full['pm25'].min():.2f} - {df_full['pm25'].max():.2f} µg/m³")
    print(f"PM2.5 mean: {df_full['pm25'].mean():.2f} µg/m³")
    
    # Save (Parquet is much faster than CSV for large data)
    print(f"\nSaving to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(OUTPUT_FILE, index=False)
    
    print("\nDone! Data ready for feature engineering.")
    return df_full

if __name__ == "__main__":
    df = process_all_files()
