"""
ERA5 Download via ecmwf-datastores (New CDS API)
=================================================
Alternative to cdsapi, uses updated infrastructure
"""

import sys
from pathlib import Path

print("=" * 70)
print("ERA5 DOWNLOAD VIA ECMWF-DATASTORES")
print("=" * 70)

try:
    from ecmwf.datasets import dataset
except ImportError:
    print("Installing ecmwf-datasets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ecmwf-datasets"])
    from ecmwf.datasets import dataset

# Ensure output directory exists
output_dir = Path(r'D:\PM25_Satellite_Research\data\raw')
output_dir.mkdir(parents=True, exist_ok=True)

print("\nDownloading ERA5 daily 2019–2023...")
print("This will take ~30–45 minutes.\n")

try:
    # Create ERA5 dataset client
    era5 = dataset("reanalysis-era5-complete")
    
    # Request configuration
    request = {
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
        'format': 'netcdf',
    }
    
    output_file = str(output_dir / 'era5_daily_2019-2023.nc')
    
    print(f"Downloading to: {output_file}\n")
    
    # Download
    era5.download(request, output_file)
    
    print(f"\n✓ Download complete! Saved to {output_file}")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nAlternative: Proceed with MERRA-2 only results")
    print("(ERA5 integration optional; MERRA-2 results already valid)")
    sys.exit(1)
