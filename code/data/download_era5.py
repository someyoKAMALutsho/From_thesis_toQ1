"""
ERA5 Daily Download (2019–2023, Global 0.5°)
==============================================
Expected download time: 30–45 minutes
File size: ~5–8 GB
"""

import cdsapi
import sys
from pathlib import Path

print("=" * 70)
print("ERA5 DAILY DATA DOWNLOAD")
print("=" * 70)
print("\nDownloading 5 years × 365 days of global ERA5 data...")
print("This will take ~30–45 minutes. Do NOT close this window.\n")

# Ensure output directory exists
output_dir = Path(r'D:\PM25_Satellite_Research\data\raw')
output_dir.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

# Request configuration
request = {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'variable': [
        'total_precipitation',
        'boundary_layer_height',
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
    ],
    'year': ['2019', '2020', '2021', '2022', '2023'],
    'month': ['01', '02', '03', '04', '05', '06', 
              '07', '08', '09', '10', '11', '12'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': '00:00',
    'grid': [0.5, 0.625],  # Match MERRA-2
    'area': [90, -180, -90, 180],  # Global
}

output_file = str(output_dir / 'era5_daily_2019-2023.nc')

print(f"Output file: {output_file}\n")
print("Submitting request to CDS server...")
print("(You can monitor progress at: https://cds.climate.copernicus.eu/requests)\n")

try:
    client.retrieve('reanalysis-era5-complete', request, output_file)
    print(f"\n{'=' * 70}")
    print(f"✓ Download complete!")
    print(f"✓ Saved to: {output_file}")
    print(f"{'=' * 70}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check your API key is correct in ~/.cdsapirc")
    print("  2. Verify internet connection")
    print("  3. Check CDS server status: https://cds.climate.copernicus.eu")
    print("  4. If request queued, check https://cds.climate.copernicus.eu/requests")
    sys.exit(1)
