"""
ERA5 Integration for Enhanced Meteorology
==========================================
MERRA-2 Gaps: Monthly only, missing PBLH/precip on some grids
ERA5 Solution: Daily resolution, complete variables

Copernicus Climate Data Store (CDS):
https://cds.climate.copernicus.eu/

Expected variables to add:
- Total precipitation (mm/day)
- Planetary boundary layer height (m, daily)
- 2m temperature (°C, daily)
- 10m wind speed (m/s, daily)

Then regrid to match MERRA-2 0.5° × 0.625° grid, aggregate to monthly.

NOTE: This is a data acquisition template; actual ERA5 download requires:
1. CDS account registration
2. CDS API setup (https://cds.climate.copernicus.eu/api-how-to)
3. ~30 min download time for 2019-2023 global daily data
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("ERA5 INTEGRATION GUIDE")
print("=" * 70)

print("""
Step 1: Register at Copernicus CDS
URL: https://cds.climate.copernicus.eu/user/register
Get API key from: https://cds.climate.copernicus.eu/api-how-to

Step 2: Install CDS API
pip install cdsapi

Step 3: Create ~/.cdsapirc file with credentials
Details: https://cds.climate.copernicus.eu/api-how-to

Step 4: Download ERA5 daily data (2019-2023)
""")

print("""
# Python script to download ERA5 (template)
import cdsapi

client = cdsapi.Client()

# Download daily ERA5 for 2019-2023
request = {
    'product_type': 'reanalysis',
    'variable': [
        'total_precipitation',
        'boundary_layer_height',
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind'
    ],
    'year': ['2019', '2020', '2021', '2022', '2023'],
    'month': ['01', '02', ..., '12'],
    'day': ['01', '02', ..., '31'],
    'time': '00:00',
    'grid': [0.5, 0.625],  # Match MERRA-2
    'area': [90, -180, -90, 180],  # Global
    'format': 'netcdf'
}

client.retrieve('reanalysis-era5-complete', request, 'era5_2019-2023.nc')
""")

print("\nStep 5: Aggregate daily to monthly (using xarray)")

print("""
import xarray as xr

# Load ERA5 daily
era5 = xr.open_dataset('era5_2019-2023.nc')

# Aggregate to monthly mean
era5_monthly = era5.resample(time='MS').mean()

# Merge with MERRA-2 dataset
# (Implementation in 13_baseline_training_GPU.py after ERA5 download)
""")

print("\nExpected R² improvements (after ERA5 integration + retraining):")
print("  Subtropical: 0.609 → ~0.65 (+0.04)")
print("  Tropical: 0.085 → ~0.15 (+0.07, major gain)")
print("  Overall: 0.341 → ~0.38 (+0.04)")

print("\n⏱️  Estimated time: 30 min setup + 30 min download + 30 min integration")
print("\nFor now, use MERRA-2 results for paper; ERA5 as future work section.")
