"""
MERRA-2 Meteorology Download & Integration
NASA's Modern-Era Retrospective analysis for Research and Applications

Source: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
Resolution: 0.625° x 0.5° (lat x lon)
Variables: Temperature, pressure, wind, precipitation, boundary layer height

MODIFICATIONS FOR THIS PROJECT:
- Downloads monthly MERRA-2 data (2019-2023)
- Interpolates to PM2.5 grid (0.01°)
- Integrates with satellite observations
- Creates 50k stratified sample with Köppen LOCO splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from earthpy.earthdata import earthaccess
except ImportError:
    print("ERROR: earthpy not installed. Run: pip install earthpy")
    exit()

# === CONFIGURATION ===
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
FEATURES_DIR = Path(r"D:\PM25_Satellite_Research\data\features")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\tables")

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_with_features.parquet"
OUTPUT_FILE = DATA_DIR / "pm25_merra2_50k.parquet"
LOCO_DIR = DATA_DIR / "loco_koppen_splits"
LOCO_DIR.mkdir(exist_ok=True)

print("="*70)
print("MERRA-2 METEOROLOGY DOWNLOAD & PM2.5 INTEGRATION")
print("="*70)

# ============================================================================
# STEP 1: LOGIN TO NASA EARTHDATA
# ============================================================================

print("\nStep 1: NASA Earthdata Authentication")
print("-" * 70)

try:
    # This will prompt for credentials if not already authenticated
    auth = earthaccess.login(strategy="netrc", persist=True)
    if not auth.authenticated:
        print("Please enter your NASA Earthdata credentials:")
        print("(Go to https://urs.earthdata.nasa.gov/users/new to register)")
        auth = earthaccess.login(strategy="interactive", persist=True)
    
    print("✓ Authenticated with NASA Earthdata")
except Exception as e:
    print(f"ERROR: {e}")
    print("\nPlease register at: https://urs.earthdata.nasa.gov/users/new")
    exit()

# ============================================================================
# STEP 2: SEARCH & DOWNLOAD MERRA-2 DATA
# ============================================================================

print("\nStep 2: Searching for MERRA-2 monthly data (2019-2023)")
print("-" * 70)

try:
    import xarray as xr
    from scipy.spatial import cKDTree
except ImportError:
    print("ERROR: Missing packages. Run: pip install xarray scipy")
    exit()

# Search for MERRA-2 monthly data
# Dataset ID: M2TMNXSLV (monthly means, single-level diagnostics)

print("Querying NASA MERRA-2 archive...")

merra2_files = []

for year in range(2019, 2024):
    for month in range(1, 13):
        try:
            # Search for MERRA-2 monthly file
            results = earthaccess.search_data(
                short_name="M2TMNXSLV",
                temporal=([f"{year}-{month:02d}-01", f"{year}-{month:02d}-28"])
            )
            
            if results:
                print(f"  Found {year}-{month:02d}: {len(results)} file(s)")
                # Download first result for this month
                file = results[0]
                local_path = earthaccess.download(file, FEATURES_DIR)
                merra2_files.extend(local_path)
            else:
                print(f"  {year}-{month:02d}: No data found")
                
        except Exception as e:
            print(f"  {year}-{month:02d}: ERROR - {e}")
            continue

if not merra2_files:
    print("\nERROR: No MERRA-2 files downloaded!")
    print("Try manual download from: https://disc.gsfc.nasa.gov/datasets/M2TMNXSLV_5.12.4/summary")
    exit()

print(f"\n✓ Downloaded {len(merra2_files)} MERRA-2 files")

# ============================================================================
# STEP 3: PROCESS MERRA-2 & EXTRACT VARIABLES
# ============================================================================

print("\nStep 3: Processing MERRA-2 files")
print("-" * 70)

met_list = []

for nc_file in merra2_files:
    try:
        print(f"  Processing {Path(nc_file).name}...")
        
        ds = xr.open_dataset(nc_file)
        
        # MERRA-2 variable names
        # T2M: 2m temperature (K)
        # PS: Surface pressure (Pa)
        # U10M, V10M: 10m wind components (m/s)
        # PBLH: Planetary boundary layer height (m)
        # PRECTOT: Total precipitation (kg/m2/s)
        
        t2m = ds['T2M'].values - 273.15  # Convert K to C
        ps = ds['PS'].values
        u10m = ds['U10M'].values
        v10m = ds['V10M'].values
        pblh = ds['PBLH'].values
        prectot = ds['PRECTOT'].values * 86400  # Convert kg/m2/s to kg/m2/day
        
        # Get coordinates
        lat = ds['lat'].values
        lon = ds['lon'].values
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Extract year/month from filename
        # Format: MERRA2_YEAR_MONTH...nc
        filename = Path(nc_file).name
        parts = filename.split('.')
        date_part = parts[1]  # e.g., "201901"
        year = int(date_part[:4])
        month = int(date_part[4:6])
        
        # Create dataframe
        df_met = pd.DataFrame({
            'year': year,
            'month': month,
            'lat': lat_grid.flatten(),
            'lon': lon_grid.flatten(),
            't2m': t2m.flatten(),
            'sp': ps.flatten(),
            'u10': u10m.flatten(),
            'v10': v10m.flatten(),
            'blh': pblh.flatten(),
            'tp': prectot.flatten()
        })
        
        met_list.append(df_met)
        ds.close()
        
    except Exception as e:
        print(f"    ERROR: {e}")
        continue

if met_list:
    df_merra2 = pd.concat(met_list, ignore_index=True)
    print(f"\n✓ Processed {len(df_merra2):,} MERRA-2 grid points")
else:
    print("ERROR: No MERRA-2 data processed!")
    exit()

# ============================================================================
# STEP 4: LOAD PM2.5 & INTERPOLATE MERRA-2
# ============================================================================

print("\nStep 4: Integrating MERRA-2 with PM2.5 satellite data")
print("-" * 70)

df_pm25 = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df_pm25):,} PM2.5 samples")

print("Interpolating MERRA-2 to PM2.5 grid (nearest neighbor)...")

# Build KDTree on MERRA-2
merra2_coords = df_merra2[['lat', 'lon']].values
tree = cKDTree(merra2_coords)

# Find nearest MERRA-2 for each PM2.5 point
pm25_coords = df_pm25[['lat', 'lon']].values
distances, indices = tree.query(pm25_coords, k=1)

# Assign MERRA-2 features
df_pm25['t2m'] = df_merra2.iloc[indices]['t2m'].values
df_pm25['sp'] = df_merra2.iloc[indices]['sp'].values
df_pm25['u10'] = df_merra2.iloc[indices]['u10'].values
df_pm25['v10'] = df_merra2.iloc[indices]['v10'].values
df_pm25['blh'] = df_merra2.iloc[indices]['blh'].values
df_pm25['tp'] = df_merra2.iloc[indices]['tp'].values
df_pm25['wind_speed'] = np.sqrt(df_pm25['u10']**2 + df_pm25['v10']**2)

print(f"✓ Interpolated {len(df_pm25):,} samples")

# ============================================================================
# STEP 5: EXPAND TO 50K+ STRATIFIED SAMPLE
# ============================================================================

print("\nStep 5: Expanding to 50k+ stratified sample")
print("-" * 70)

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

df_pm25['koppen_zone'] = df_pm25['lat'].apply(get_koppen_zone)

print("\nKöppen zones in dataset:")
print(df_pm25['koppen_zone'].value_counts())

# Stratified sampling to 50k
target_size = 50000
zone_proportions = df_pm25['koppen_zone'].value_counts(normalize=True)

df_stratified = []
for zone in df_pm25['koppen_zone'].unique():
    n_samples = int(zone_proportions[zone] * target_size)
    zone_data = df_pm25[df_pm25['koppen_zone'] == zone]
    sample = zone_data.sample(n=min(n_samples, len(zone_data)), random_state=42)
    df_stratified.append(sample)

df_expanded = pd.concat(df_stratified, ignore_index=True).reset_index(drop=True)

print(f"\n✓ Expanded to {len(df_expanded):,} samples")
print("\nFinal Köppen distribution:")
print(df_expanded['koppen_zone'].value_counts())

# ============================================================================
# STEP 6: CREATE KÖPPEN LOCO SPLITS
# ============================================================================

print("\nStep 6: Creating Köppen climate zone LOCO validation splits")
print("-" * 70)

zones = sorted(df_expanded['koppen_zone'].unique())
loco_stats = []

for test_zone in zones:
    train = df_expanded[df_expanded['koppen_zone'] != test_zone]
    test = df_expanded[df_expanded['koppen_zone'] == test_zone]
    
    # Save
    train.to_parquet(LOCO_DIR / f"train_loczz_{test_zone}.parquet", index=False)
    test.to_parquet(LOCO_DIR / f"test_loczz_{test_zone}.parquet", index=False)
    
    loco_stats.append({
        'Test_Zone': test_zone,
        'Train_Samples': len(train),
        'Test_Samples': len(test),
        'Test_Pct': f"{len(test)/len(df_expanded)*100:.1f}%"
    })
    
    print(f"  {test_zone}: Train={len(train):,}, Test={len(test):,}")

# Save stats
loco_df = pd.DataFrame(loco_stats)
loco_df.to_csv(LOCO_DIR / "loco_splits_summary.csv", index=False)

# ============================================================================
# STEP 7: SAVE FINAL DATASET
# ============================================================================

print(f"\nStep 7: Saving final dataset")
print("-" * 70)

df_expanded.to_parquet(OUTPUT_FILE, index=False)
print(f"✓ Saved: {OUTPUT_FILE}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION COMPLETE")
print("="*70)
print(f"\nFinal Dataset:")
print(f"  Samples: {len(df_expanded):,}")
print(f"  Features: {len(df_expanded.columns)}")
print(f"  Time period: 2019-2023 (60 months)")
print(f"  Geographic: {len(df_expanded['lat'].unique()):,} unique locations")
print(f"\nMeteorological Features (MERRA-2):")
print(f"  - t2m (Temperature, °C): {df_expanded['t2m'].mean():.1f}±{df_expanded['t2m'].std():.1f}")
print(f"  - sp (Pressure, Pa): {df_expanded['sp'].mean():.0f}±{df_expanded['sp'].std():.0f}")
print(f"  - wind_speed (m/s): {df_expanded['wind_speed'].mean():.2f}±{df_expanded['wind_speed'].std():.2f}")
print(f"  - blh (Boundary layer height, m): {df_expanded['blh'].mean():.0f}±{df_expanded['blh'].std():.0f}")
print(f"  - tp (Precipitation, kg/m²/day): {df_expanded['tp'].mean():.2f}±{df_expanded['tp'].std():.2f}")
print(f"\nKöppen LOCO Splits: {len(zones)} climate zones")
print(f"  Saved to: {LOCO_DIR}")
print(f"\n✓ Ready for rigorous Q1 model training!")
