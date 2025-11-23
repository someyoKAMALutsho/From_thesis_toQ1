import earthaccess
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURABLES ---
OUTDIR = Path(r'D:\PM25_Satellite_Research\data\merra2_hourly')
OUTDIR.mkdir(parents=True, exist_ok=True)
AGG_FILE_CSV = OUTDIR / "monthly_meteorology_merra2_2019_2023.csv"
AGG_FILE_PARQUET = OUTDIR / "monthly_meteorology_merra2_2019_2023.parquet"

years = range(2019, 2024)
months = range(1, 13)
variables = ["T2M", "PS", "U10M", "V10M", "PBLTOP", "PRECTOT"]

print("=" * 60)
print("NASA MERRA-2 Hourly (M2T1NXSLV) to Monthly Aggregation")
print("=" * 60)

# 1. NASA Login
earthaccess.login()

def download_hourly_for_month(year, month, outdir):
    """Download all hourly files for this month/year, if not present."""
    ym_str = f"{year}{month:02d}"
    product = "M2T1NXSLV"  # hourly, single level
    print(f"  Searching hourly MERRA-2 {product} for {ym_str}")
    results = earthaccess.search_data(
        short_name=product,
        temporal=(f"{year}-{month:02d}-01", f"{year}-{month:02d}-28")
    )
    files = []
    for i, granule in enumerate(results):
        # Try multiple ways to get filename
        try:
            # Method 1: from data_links
            fname = granule.data_links()[0].split('/')[-1] if granule.data_links() else None
        except:
            fname = None
        
        if fname is None:
            # Method 2: just use a generic name
            fname = f"merra2_hourly_{year}{month:02d}_{i:03d}.nc4"
        
        outf = outdir / fname
        if outf.exists():
            print(f"    Found existing {outf.name}")
        else:
            print(f"    Downloading {fname}")
            earthaccess.download(granule, outdir)
        files.append(outf)
    return files

def extract_and_agg_monthly(ncfiles, vars=variables):
    """Aggregate monthly mean for every lat/lon/var."""
    monthly_arrays = []
    for ncf in tqdm(ncfiles, desc="Aggregating"):
        if not ncf.exists():
            print(f"WARNING: {ncf.name} not found, skipping")
            continue
        with xr.open_dataset(ncf) as ds:
            if not {"lat", "lon", "time"}.issubset(ds.variables):
                print(f"WARNING: {ncf} does not have expected grid, skipping.")
                continue
            arrs = {}
            for var in vars:
                arr = ds[var].values if var in ds else None
                if arr is None:
                    print(f"  WARNING: {var} not found in {ncf.name}, filled with NaN.")
                    arrs[var] = np.full((ds.dims["lat"], ds.dims["lon"]), np.nan)
                elif arr.ndim == 3:
                    arrs[var] = np.nanmean(arr, axis=0)  # mean over time axis
                elif arr.ndim == 2:
                    arrs[var] = arr  # already single time step
                else:
                    arrs[var] = np.full((ds.dims["lat"], ds.dims["lon"]), np.nan)
            # Extract date from filename or ds
            try:
                fname = ncf.name
                date_cands = [seg for seg in fname.split(".") if seg.isdigit() and len(seg) == 8]
                if date_cands:
                    year = int(date_cands[0][:4])
                    month = int(date_cands[0][4:6])
                else:
                    # Use ds["time"].values[0]
                    datestr = str(pd.to_datetime(ds["time"].values[0])).split("T")[0]
                    year, month = int(datestr[:4]), int(datestr[5:7])
            except:
                year, month = None, None
            lon, lat = ds["lon"].values, ds["lat"].values
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            for i in range(lon_grid.shape[0]):
                for j in range(lon_grid.shape[1]):
                    rec = {
                        "year": year, "month": month,
                        "lat": lat_grid[i, j], "lon": lon_grid[i, j]
                    }
                    for var in vars:
                        rec[var] = arrs[var][i, j]
                    monthly_arrays.append(rec)
    return pd.DataFrame(monthly_arrays)

# 2. Download and process
all_nc_files = []
for year in years:
    for month in months:
        newfiles = download_hourly_for_month(year, month, OUTDIR)
        all_nc_files.extend(newfiles)

print("\nAggregating all months to monthly means...")
df_agg = extract_and_agg_monthly(all_nc_files, vars=variables)
print(f"\nSaving to {AGG_FILE_CSV} and {AGG_FILE_PARQUET} ...")
df_agg.to_csv(AGG_FILE_CSV, index=False)
df_agg.to_parquet(AGG_FILE_PARQUET, index=False)

print("\n==== SUCCESS ====")
print(f"All-month aggregated meteorology saved as:\n- {AGG_FILE_CSV}\n- {AGG_FILE_PARQUET}")
print("You can now merge these files by year/month/lat/lon with your PM2.5 data.")
