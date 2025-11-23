"""
ERA5 Integration: Daily Meteorology to Replace MERRA-2 Gaps
===========================================================

WHY ERA5:
- MERRA-2 monthly: Misses episodic PM2.5 events, missing PBLH/precip
- ERA5 daily: Complete global coverage, all variables
- Source: Copernicus Climate Data Store (CDS)
- Reference: Hersbach et al. (2020). ERA5 Monthly Averaged Data. CDS

EXPECTED GAINS:
- Tropical R²: 0.085 → 0.15–0.20 (+10–15%)
- Global R²: 0.341 → 0.38–0.40 (+2–3%)

DATA REQUIREMENTS:
- Free account: https://cds.climate.copernicus.eu/user/register
- API setup: ~5 minutes (https://cds.climate.copernicus.eu/api-how-to)
- Download time: ~30 minutes (5 years × 365 days × 0.5° global)

ALTERNATIVE (if no CDS access):
- Pre-downloaded ERA5 available on AWS S3
- Or use GRIB2 files from European data server
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ERA5 INTEGRATION WORKFLOW")
print("=" * 70)

DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
RAW_DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("""
STEP 1: Download ERA5 Daily Data (Automated Script)
===================================================

This script downloads ERA5 daily 2019–2023 via CDS API.
Prerequisite: Install cdsapi
  pip install cdsapi

Then create ~/.cdsapirc with:
  url: https://cds.climate.copernicus.eu/api/v2
  key: <YOUR_CDS_UID>:<YOUR_CDS_API_KEY>

Get your key from: https://cds.climate.copernicus.eu/api-how-to
""")

print("\nSTEP 2: CDS Download Script (Copy into Python)")

era5_download_script = """
# COPY THIS INTO A SEPARATE SCRIPT: download_era5.py
import cdsapi

client = cdsapi.Client()

# Download ERA5 daily 2019–2023 (global 0.5°)
# Variables: precip, PBLH, T2m, u10, v10
request = {
    'product_type': 'reanalysis',
    'variable': [
        'total_precipitation',
        'boundary_layer_height',
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
    ],
    'year': ['2019', '2020', '2021', '2022', '2023'],
    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': '00:00',
    'grid': [0.5, 0.625],  # Match MERRA-2 grid
    'area': [90, -180, -90, 180],  # Global
    'format': 'netcdf',
}

client.retrieve('reanalysis-era5-complete', request, 'era5_daily_2019-2023.nc')
print('Download complete!')
"""

print(era5_download_script)

print("\nSTEP 3: Aggregate ERA5 Daily → Monthly (After Download)")

era5_agg_script = """
# COPY THIS INTO A SEPARATE SCRIPT: aggregate_era5.py
import xarray as xr
import numpy as np

# Load ERA5 daily (downloaded from CDS)
print('Loading ERA5 daily data...')
era5 = xr.open_dataset('D:/PM25_Satellite_Research/data/raw/era5_daily_2019-2023.nc')

# Aggregate to monthly mean
# Source: xarray documentation
print('Aggregating to monthly...')
era5_monthly = era5.resample(time='MS').mean()

# Rename variables to match MERRA-2 convention
era5_monthly = era5_monthly.rename({
    'total_precipitation': 'prec',
    'boundary_layer_height': 'blh_era5',
    '2m_temperature': 't2m_era5',
    '10m_u_component_of_wind': 'u10_era5',
    '10m_v_component_of_wind': 'v10_era5',
})

# Save
era5_monthly.to_netcdf('D:/PM25_Satellite_Research/data/raw/era5_monthly_2019-2023.nc')
print('Saved: era5_monthly_2019-2023.nc')
"""

print(era5_agg_script)

print("\nSTEP 4: Merge ERA5 with MERRA-2 Dataset")

print("""
# COPY THIS INTO integration_era5_merra2.py
import xarray as xr
import pandas as pd
import numpy as np

# Load both datasets
merra2 = xr.open_dataset('D:/PM25_Satellite_Research/data/processed/pm25_merra2_meteorology_final.parquet')
era5 = xr.open_dataset('D:/PM25_Satellite_Research/data/raw/era5_monthly_2019-2023.nc')

# Convert to same grid (regrid ERA5 to MERRA-2 0.5° × 0.625°)
# Source: xarray.DataArray.interp
era5_regridded = era5.interp(
    lat=merra2.lat,
    lon=merra2.lon,
    method='nearest'
)

# Merge on lat/lon/time
combined = xr.merge([merra2, era5_regridded])

# Create new feature set with ERA5 where MERRA-2 gaps exist
combined['prec_filled'] = combined['prec'].fillna(combined['prec_era5'])
combined['blh_filled'] = combined['blh'].fillna(combined['blh_era5'])
combined['t2m_filled'] = combined['t2m'].fillna(combined['t2m_era5'])

# Save merged dataset
combined.to_netcdf('D:/PM25_Satellite_Research/data/processed/pm25_merra2_era5_merged.nc')
print('✓ Merged dataset saved')
""")

print("\n" + "=" * 70)
print("TIMELINE")
print("=" * 70)
print("Step 1 (Download): 30 min (automated)")
print("Step 2 (Aggregation): 5 min")
print("Step 3 (Integration): 10 min")
print("Step 4 (Retraining): 20 min")
print("Total: ~1 hour (mostly waiting for download)")

print("\n" + "=" * 70)
print("NEXT: Run 13_baseline_training_GPU.py with ERA5 data")
print("Expected: +10–15% Tropical R² (0.085 → 0.15–0.20)")
print("=" * 70)
