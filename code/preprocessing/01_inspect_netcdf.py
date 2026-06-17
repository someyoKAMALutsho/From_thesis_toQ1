"""
Quick NetCDF structure inspection to identify variable names.
Adapted for Washington University SAT PM2.5 dataset.
"""

import xarray as xr
from pathlib import Path

# Path to your NetCDF files
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
nc_dir      = PROJECT_DIR / "data" / "raw" / "monthly_netcdf" / "2019"

# Get first .nc file
nc_files = list(nc_dir.glob("*.nc"))
if not nc_files:
    print("No .nc files found!")
    exit()

sample_file = nc_files[0]
print(f"Inspecting: {sample_file.name}\n")

# Open and inspect
ds = xr.open_dataset(sample_file)
print("=== VARIABLES ===")
print(list(ds.variables))

print("\n=== DIMENSIONS ===")
print(ds.dims)

print("\n=== DATA VARIABLES (with shapes) ===")
for var in ds.data_vars:
    print(f"{var}: {ds[var].shape}")

ds.close()
