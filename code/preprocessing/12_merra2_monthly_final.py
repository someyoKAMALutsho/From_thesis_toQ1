"""
MERRA-2 Monthly Meteorology Download & PM2.5 Integration
Product: M2TMNXSLV (tavgM_2d_slv_Nx) - Monthly Means, Single Level
Downloads 60 files (2019-2023), extracts key meteorology variables,
and merges with PM2.5 satellite data for Q1-ready analysis.

Output files:
- pm25_merra2_meteorology_final.parquet (merged PM2.5 + meteorology)
- loco_koppen_splits/ (Köppen climate zone LOCO splits)
"""

import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import time

DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
FEATURES_DIR = Path(r"D:\PM25_Satellite_Research\data\merra2_monthly")
OUTPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"
LOCO_DIR = DATA_DIR / "loco_koppen_splits_final"
LOCO_DIR.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_with_features.parquet"

def get_koppen_zone(lat):
    """Simplified Köppen climate classification."""
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        return 'Tropical'
    elif 23.5 < abs_lat <= 35:
        return 'Subtropical'
    elif 35 < abs_lat <= 50:
        return 'Temperate'
    elif 50 < abs_lat <= 60:
        return 'Boreal'
    else:
        return 'Polar'

def safe_extract(ds, var_names, fallback_shape):
    """
    Try to extract variable from list of possible names.
    Returns (array, found_name) or (zeros_array, None) if not found.
    """
    for var_name in var_names:
        if var_name in ds:
            try:
                arr = ds[var_name].values
                # Handle scalar
                if arr.ndim == 0:
                    arr = np.full(fallback_shape, arr)
                # Handle 2D
                elif arr.ndim == 2:
                    if arr.shape != fallback_shape:
                        arr = np.broadcast_to(arr, fallback_shape)
                # Handle 3D (time dimension)
                elif arr.ndim == 3:
                    arr = np.nanmean(arr, axis=0)
                    if arr.shape != fallback_shape:
                        arr = np.broadcast_to(arr, fallback_shape)
                else:
                    arr = np.full(fallback_shape, np.nan)
                return arr, var_name
            except Exception as e:
                print(f"  WARNING: Error extracting {var_name}: {e}")
                continue
    # Not found
    return np.full(fallback_shape, np.nan), None

print("=" * 70)
print("MERRA-2 Monthly Meteorology (M2TMNXSLV) Download & PM2.5 Merge")
print("=" * 70)

print("\nStep 1: NASA Earthdata Login...")
earthaccess.login()
print("✓ Authenticated!")

# Download monthly MERRA-2 (60 files: 2019-2023)
print("\nStep 2: Downloading 60 monthly MERRA-2 files (M2TMNXSLV)...")
years = range(2019, 2024)
months = range(1, 13)
all_files = []

for year in years:
    for month in months:
        ym_str = f"{year}-{month:02d}"
        print(f"  {ym_str}...", end=" ", flush=True)
        results = earthaccess.search_data(
            short_name="M2TMNXSLV",
            temporal=(f"{year}-{month:02d}-01", f"{year}-{month:02d}-28")
        )
        if results:
            for granule in results:
                try:
                    # Try to get filename
                    links = granule.data_links()
                    fname = links[0].split('/')[-1] if links else f"merra2_{ym_str}.nc4"
                except:
                    fname = f"merra2_{ym_str}.nc4"
                
                outf = FEATURES_DIR / fname
                if outf.exists():
                    print(f"(cached)", end=" ")
                else:
                    earthaccess.download(granule, FEATURES_DIR)
                all_files.append(outf)
            print("✓")
        else:
            print("(not found)")

print(f"\n✓ Total monthly MERRA-2 files: {len(all_files)}")

# Process and extract meteorology
print("\nStep 3: Extracting meteorology variables from NetCDFs...")
met_list = []

for nc_file in sorted(FEATURES_DIR.glob("*.nc4")):
    if not nc_file.exists():
        continue
    
    print(f"  Processing {nc_file.name}...")
    ds = xr.open_dataset(nc_file)
    
    # Verify grid
    if "lat" not in ds or "lon" not in ds:
        print(f"    WARNING: No lat/lon grid, skipping")
        ds.close()
        continue
    
    lat = ds["lat"].values
    lon = ds["lon"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    fallback_shape = lon_grid.shape
    
    # Extract key meteorology variables
    # For monthly product, we use available variables
    t2m, t2m_name = safe_extract(ds, ["T2M"], fallback_shape)
    if t2m_name:
        t2m = t2m - 273.15  # K to C
    
    ps, ps_name = safe_extract(ds, ["PS"], fallback_shape)
    u10m, u10m_name = safe_extract(ds, ["U10M"], fallback_shape)
    v10m, v10m_name = safe_extract(ds, ["V10M"], fallback_shape)
    pblh, pblh_name = safe_extract(ds, ["PBLH", "PBLTOP"], fallback_shape)
    prectot, prectot_name = safe_extract(ds, ["PRECTOT"], fallback_shape)
    if prectot_name:
        prectot = prectot * 86400  # kg/m2/s to kg/m2/day
    
    # Extract date
    try:
        fname = nc_file.name
        date_cands = [seg for seg in fname.split(".") if seg.isdigit() and len(seg) == 6]
        if date_cands:
            year = int(date_cands[0][:4])
            month = int(date_cands[0][4:6])
        else:
            # Try 8-digit YYYYMMDD
            date_cands = [seg for seg in fname.split(".") if seg.isdigit() and len(seg) == 8]
            if date_cands:
                year = int(date_cands[0][:4])
                month = int(date_cands[0][4:6])
            else:
                print(f"    WARNING: Cannot parse date, skipping")
                ds.close()
                continue
    except Exception as e:
        print(f"    WARNING: Date parsing error: {e}")
        ds.close()
        continue
    
    df_met = pd.DataFrame({
        "year": year,
        "month": month,
        "lat": lat_grid.flatten(),
        "lon": lon_grid.flatten(),
        "t2m": t2m.flatten(),
        "ps": ps.flatten(),
        "u10": u10m.flatten(),
        "v10": v10m.flatten(),
        "blh": pblh.flatten(),
        "prec": prectot.flatten()
    })
    met_list.append(df_met)
    ds.close()

df_met_all = pd.concat(met_list, ignore_index=True)
print(f"✓ Processed {len(df_met_all):,} meteorology grid points")

# Load PM2.5 and merge
print("\nStep 4: Loading PM2.5 satellite data...")
df_pm25 = pd.read_parquet(INPUT_FILE)
print(f"  {len(df_pm25):,} PM2.5 samples loaded")

print("Step 5: Interpolating MERRA-2 to PM2.5 grid (nearest neighbor)...")
tree = cKDTree(df_met_all[["lat", "lon"]].values)
pm25_coords = df_pm25[["lat", "lon"]].values
_, idxs = tree.query(pm25_coords, k=1)

for col in ["t2m", "ps", "u10", "v10", "blh", "prec"]:
    df_pm25[col] = df_met_all.iloc[idxs][col].values

df_pm25["wind_speed"] = np.sqrt(df_pm25["u10"]**2 + df_pm25["v10"]**2)
print(f"✓ Interpolated meteorology to {len(df_pm25):,} PM2.5 samples")

# Assign Köppen zones
print("\nStep 6: Assigning Köppen climate zones...")
df_pm25['koppen_zone'] = df_pm25['lat'].apply(get_koppen_zone)
print(df_pm25['koppen_zone'].value_counts())

# Stratified sampling
print("\nStep 7: Stratified sampling to 50,000 examples...")
zone_props = df_pm25['koppen_zone'].value_counts(normalize=True)
samples = []
for zone in df_pm25['koppen_zone'].unique():
    n = int(zone_props[zone] * 50000)
    zone_data = df_pm25[df_pm25['koppen_zone'] == zone]
    samples.append(zone_data.sample(n=min(n, len(zone_data)), random_state=42))

df_expanded = pd.concat(samples, ignore_index=True).reset_index(drop=True)
print(f"Final sample: {len(df_expanded):,}")

# Create LOCO splits
print("\nStep 8: Creating Köppen LOCO validation splits...")
for zone in sorted(df_expanded['koppen_zone'].unique()):
    train = df_expanded[df_expanded['koppen_zone'] != zone]
    test = df_expanded[df_expanded['koppen_zone'] == zone]
    train.to_parquet(LOCO_DIR / f"train_loczz_{zone}.parquet", index=False)
    test.to_parquet(LOCO_DIR / f"test_loczz_{zone}.parquet", index=False)
    print(f"  {zone}: Train={len(train):,}, Test={len(test):,}")

# Save final dataset
print(f"\nStep 9: Saving final dataset: {OUTPUT_FILE}")
df_expanded.to_parquet(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("SUCCESS! Q1-READY DATASET COMPLETE")
print("=" * 70)
print(f"\nFinal Dataset Summary:")
print(f"  Total samples: {len(df_expanded):,}")
print(f"  Time period: 2019-2023 (60 months)")
print(f"  Unique locations: {len(df_expanded['lat'].unique()):,}")
print(f"  Meteorology variables: t2m, ps, u10, v10, blh, prec, wind_speed")
print(f"  Köppen zones: {len(df_expanded['koppen_zone'].unique())}")
print(f"\nOutput files:")
print(f"  - {OUTPUT_FILE}")
print(f"  - {LOCO_DIR}/ (LOCO splits)")
print(f"\n✓ Ready for ML model training, baseline comparison, and paper writing!")
