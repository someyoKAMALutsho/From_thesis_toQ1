"""
MERRA-2 Monthly Meteorology Download & PM2.5 Integration
Product: M2TMNXSLV (tavgM_2d_slv_Nx) - Monthly Means, Single Level

Downloads 60 files (2019–2023), extracts key meteorology variables,
and merges them with PM2.5 satellite data for Q1-ready analysis.

Output files:
- pm25_merra2_meteorology_full.parquet   (full merged PM2.5 + meteorology, ~1.05M rows)
- pm25_merra2_meteorology_final.parquet  (stratified ~50k sample used in the paper)
- loco_koppen_splits_final/              (Köppen climate zone LOCO splits from the sample)
"""

import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR     = PROJECT_DIR / "data" / "processed"
FEATURES_DIR = PROJECT_DIR / "data" / "merra2_monthly"
OUTPUT_FILE  = DATA_DIR / "pm25_merra2_meteorology_final.parquet"
FULL_FILE    = DATA_DIR / "pm25_merra2_meteorology_full.parquet"
LOCO_DIR     = DATA_DIR / "loco_koppen_splits_final"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOCO_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE   = PROJECT_DIR / "data" / "processed" / "pm25_with_features.parquet"



def get_koppen_zone(lat: float) -> str:
    """Simplified Köppen climate classification based on absolute latitude."""
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        return "Tropical"
    elif 23.5 < abs_lat <= 35:
        return "Subtropical"
    elif 35 < abs_lat <= 50:
        return "Temperate"
    elif 50 < abs_lat <= 60:
        return "Boreal"
    else:
        return "Polar"


def safe_extract(ds, var_names, fallback_shape):
    """
    Try to extract a variable from a list of possible names.
    Returns (array, found_name) or (nan_array, None) if not found.
    """
    for var_name in var_names:
        if var_name in ds:
            try:
                arr = ds[var_name].values
                # Scalar
                if arr.ndim == 0:
                    arr = np.full(fallback_shape, arr)
                # 2D
                elif arr.ndim == 2:
                    if arr.shape != fallback_shape:
                        arr = np.broadcast_to(arr, fallback_shape)
                # 3D (time dimension): average over time
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
    return np.full(fallback_shape, np.nan), None


print("=" * 70)
print("MERRA-2 Monthly Meteorology (M2TMNXSLV) Download & PM2.5 Merge")
print("=" * 70)

print("\nStep 1: NASA Earthdata Login...")
earthaccess.login()
print("✓ Authenticated!")

# ---------------------------------------------------------------------------
# Step 2: Ensure monthly MERRA-2 files exist (download missing, reuse cached)
# ---------------------------------------------------------------------------
print("\nStep 2: Ensuring 60 monthly MERRA-2 files (M2TMNXSLV) are available...")
years  = range(2019, 2024)
months = range(1, 13)
all_files = []

for year in years:
    for month in months:
        ym_str = f"{year}{month:02d}"
        # Expected filename pattern (MERRA2_4xx.tavgM_2d_slv_Nx.YYYYMM.nc4)
        pattern = f"MERRA2_4*.tavgM_2d_slv_Nx.{ym_str}.nc4"
        existing = sorted(FEATURES_DIR.glob(pattern))
        if existing:
            all_files.extend(existing)
            print(f"  {year}-{month:02d} (cached)")
            continue

        print(f"  {year}-{month:02d} (missing, downloading)...", end=" ", flush=True)
        results = earthaccess.search_data(
            short_name="M2TMNXSLV",
            temporal=(f"{year}-{month:02d}-01", f"{year}-{month:02d}-28")
        )
        if not results:
            print("(not found)")
            continue

        for granule in results:
            links = granule.data_links()
            fname = links[0].split("/")[-1] if links else f"MERRA2_unknown_{ym_str}.nc4"
            outf = FEATURES_DIR / fname
            if not outf.exists():
                earthaccess.download(granule, FEATURES_DIR)
            all_files.append(outf)
        print("✓")

print(f"\n✓ Total monthly MERRA-2 files used: {len(all_files)}")

# ---------------------------------------------------------------------------
# Step 3: Extract meteorology from NetCDFs
# ---------------------------------------------------------------------------
print("\nStep 3: Extracting meteorology variables from NetCDFs...")
met_list = []

for nc_file in sorted(set(all_files)):
    if not nc_file.exists():
        continue

    print(f"  Processing {nc_file.name}...")
    try:
        ds = xr.open_dataset(nc_file)
    except Exception as e:
        print(f"    WARNING: Cannot open {nc_file.name}: {e}, skipping")
        continue

    if "lat" not in ds or "lon" not in ds:
        print("    WARNING: No lat/lon grid, skipping")
        ds.close()
        continue

    lat = ds["lat"].values
    lon = ds["lon"].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    fallback_shape = lon_grid.shape

    # Extract key meteorology variables
    t2m, t2m_name = safe_extract(ds, ["T2M"], fallback_shape)
    if t2m_name is not None:
        t2m = t2m - 273.15  # K to °C

    ps,    ps_name   = safe_extract(ds, ["PS"], fallback_shape)
    u10m,  u10m_name = safe_extract(ds, ["U10M"], fallback_shape)
    v10m,  v10m_name = safe_extract(ds, ["V10M"], fallback_shape)
    pblh,  pblh_name = safe_extract(ds, ["PBLH", "PBLTOP"], fallback_shape)
    prectot, prec_name = safe_extract(ds, ["PRECTOT"], fallback_shape)
    if prec_name is not None:
        prectot = prectot * 86400.0  # kg/m2/s to kg/m2/day

    # Extract year/month from filename
    try:
        fname = nc_file.name
        date_cands = [seg for seg in fname.split(".") if seg.isdigit() and len(seg) == 6]
        if date_cands:
            year = int(date_cands[0][:4])
            month = int(date_cands[0][4:6])
        else:
            date_cands = [seg for seg in fname.split(".") if seg.isdigit() and len(seg) == 8]
            if date_cands:
                year = int(date_cands[0][:4])
                month = int(date_cands[0][4:6])
            else:
                print("    WARNING: Cannot parse date, skipping")
                ds.close()
                continue
    except Exception as e:
        print(f"    WARNING: Date parsing error: {e}")
        ds.close()
        continue

    df_met = pd.DataFrame({
        "year":  year,
        "month": month,
        "lat":   lat_grid.ravel(),
        "lon":   lon_grid.ravel(),
        "t2m":   t2m.ravel(),
        "ps":    ps.ravel(),
        "u10":   u10m.ravel(),
        "v10":   v10m.ravel(),
        "blh":   pblh.ravel(),
        "prec":  prectot.ravel()
    })
    met_list.append(df_met)
    ds.close()

df_met_all = pd.concat(met_list, ignore_index=True)
print(f"✓ Processed {len(df_met_all):,} meteorology grid points")

# Step 4: Load PM2.5 and merge
# ---------------------------------------------------------------------------
print("\nStep 4: Loading PM2.5 satellite data...")
df_pm25 = pd.read_parquet(INPUT_FILE)
print(f"  {len(df_pm25):,} PM2.5 samples loaded")

print("Step 5: Interpolating MERRA-2 to PM2.5 grid (nearest neighbour)...")
tree = cKDTree(df_met_all[["lat", "lon"]].values)
pm25_coords = df_pm25[["lat", "lon"]].values
_, idxs = tree.query(pm25_coords, k=1)

for col in ["t2m", "ps", "u10", "v10", "blh", "prec"]:
    df_pm25[col] = df_met_all.iloc[idxs][col].values

print(f"✓ Interpolated meteorology to {len(df_pm25):,} PM2.5 samples")

# ---------------------------------------------------------------------------
# Step 6: Assign Köppen zones
# ---------------------------------------------------------------------------
print("\nStep 6: Assigning Köppen climate zones...")
df_pm25["koppen_zone"] = df_pm25["lat"].apply(get_koppen_zone)
print(df_pm25["koppen_zone"].value_counts())

# ---------------------------------------------------------------------------
# Step 7: Save full dataset (for diagnostics and KS tests)
# ---------------------------------------------------------------------------
print(f"\nStep 7: Saving full merged dataset: {FULL_FILE}")
df_pm25.to_parquet(FULL_FILE, index=False)

# ---------------------------------------------------------------------------
# Step 8: Stratified sampling to ~50,000 examples
# ---------------------------------------------------------------------------
print("\nStep 8: Stratified sampling to 50,000 examples...")
zone_props = df_pm25["koppen_zone"].value_counts(normalize=True)
samples = []
for zone in df_pm25["koppen_zone"].unique():
    n = int(zone_props[zone] * 50000)
    zone_data = df_pm25[df_pm25["koppen_zone"] == zone]
    samples.append(zone_data.sample(n=min(n, len(zone_data)), random_state=42))

df_expanded = pd.concat(samples, ignore_index=True).reset_index(drop=True)
print(f"Final sample: {len(df_expanded):,}")

# If you want to drop wind_speed from the modelling feature set, do it here:
if "wind_speed" in df_expanded.columns:
    df_expanded = df_expanded.drop(columns=["wind_speed"])

# ---------------------------------------------------------------------------
# Step 9: Create LOCO splits from the sample
# ---------------------------------------------------------------------------
print("\nStep 9: Creating Köppen LOCO validation splits...")
for zone in sorted(df_expanded["koppen_zone"].unique()):
    train = df_expanded[df_expanded["koppen_zone"] != zone]
    test  = df_expanded[df_expanded["koppen_zone"] == zone]
    train.to_parquet(LOCO_DIR / f"train_loczz_{zone}.parquet", index=False)
    test.to_parquet(LOCO_DIR / f"test_loczz_{zone}.parquet", index=False)
    print(f"  {zone}: Train={len(train):,}, Test={len(test):,}")

# ---------------------------------------------------------------------------
# Step 10: Save final dataset
# ---------------------------------------------------------------------------
print(f"\nStep 10: Saving final dataset: {OUTPUT_FILE}")
df_expanded.to_parquet(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("SUCCESS! Q1-READY DATASET COMPLETE")
print("=" * 70)
print(f"\nFinal Dataset Summary:")
print(f"  Total samples: {len(df_expanded):,}")
print("  Time period: 2019–2023 (60 months)")
print(f"  Unique locations: {len(df_expanded['lat'].unique()):,}")
print("  Meteorology variables: t2m, ps, u10, v10, blh, prec")
print(f"  Köppen zones: {len(df_expanded['koppen_zone'].unique())}")
print("\nOutput files:")
print(f"  - {FULL_FILE}")
print(f"  - {OUTPUT_FILE}")
print(f"  - {LOCO_DIR} (LOCO splits)")
print("\n✓ Ready for ML model training, baseline comparison, and paper writing!")
