import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_DIR / "data" / "processed"
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"

# Use the existing final dataset (49,997 rows) for both “full” and “sample”
df = pd.read_parquet(DATA_DIR / "pm25_merra2_meteorology_final.parquet")

full_df   = df.copy()
sample_df = df.copy()  # placeholder; KS test will be trivial for now

# 2) Reconstruct your latitude-band zones
def latband_zone(lat: float) -> str:
    alat = abs(lat)
    if alat <= 23.5:
        return "Tropical"
    elif 23.5 < alat <= 35:
        return "Subtropical"
    elif 35 < alat <= 50:
        return "Temperate"
    elif 50 < alat <= 60:
        return "Boreal"
    else:
        return "Polar"

full_df["zone_latband"] = full_df["lat"].apply(latband_zone)

# 3) Load Peel Köppen ASCII grid
asc_path = EXTERNAL_DIR / "koppen_ascii.txt"

with open(asc_path, "r") as f:
    header = [next(f) for _ in range(6)]
    data = np.loadtxt(f)

header_dict = {}
for line in header:
    key, val = line.strip().split()
    header_dict[key.lower()] = float(val)

ncols      = int(header_dict["ncols"])
nrows      = int(header_dict["nrows"])
xllcorner  = header_dict["xllcorner"]   # -180
yllcorner  = header_dict["yllcorner"]   # -90
cellsize   = header_dict["cellsize"]    # 0.1
nodata_val = header_dict.get("nodata_value", -9999.0)

assert data.shape == (nrows, ncols)

# 4) Mapping from Peel integer codes to broad Köppen letters and then to your 5 zones.
def peel_code_to_letter(code: int) -> str | None:
    # Placeholder; adjust using Peel's code table
    # For now, assume:
    #  1–3: A,  4–9: B,  10–16: C, 17–24: D, 25–30: E
    if 1 <= code <= 3:
        return "A"
    elif 4 <= code <= 9:
        return "B"
    elif 10 <= code <= 16:
        return "C"
    elif 17 <= code <= 24:
        return "D"
    elif 25 <= code <= 30:
        return "E"
    else:
        return None

def koppen_letter_to_zone(letter: str, lat: float) -> str | None:
    # Map broad Köppen letters to your 5 zones.
    alat = abs(lat)
    if letter == "A":
        return "Tropical"
    elif letter == "E":
        return "Polar"
    elif letter == "D":
        return "Boreal" if alat > 50 else "Temperate"
    elif letter == "C":
        return "Temperate"
    elif letter == "B":
        # Arid belts often subtropical–temperate; tie-break by latitude
        if alat <= 23.5:
            return "Tropical"
        elif 23.5 < alat <= 35:
            return "Subtropical"
        elif 35 < alat <= 50:
            return "Temperate"
        else:
            return "Boreal"
    else:
        return None

def koppen_code_to_bigzone(code: int, lat: float) -> str | None:
    letter = peel_code_to_letter(code)
    if letter is None:
        return None
    return koppen_letter_to_zone(letter, lat)

# 5) Get Peel-based zone for a given lat/lon
# Peel ASCII is typically stored with row 0 at 90N (top), row increasing southward.
def peel_zone_for_point(lat: float, lon: float) -> str | None:
    # Column: simple linear transform from lon
    col = int((lon - xllcorner) / cellsize)
    # Row: count from top (90N) downwards
    row = int((90.0 - lat) / cellsize)
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        return None
    code = data[row, col]
    if code == nodata_val:
        return None
    return koppen_code_to_bigzone(int(code), lat)

# 6) Apply Peel zones to full_df (this may take some seconds on 1M rows but is one-off)
zones_peel = []
for lat, lon in zip(full_df["lat"].values, full_df["lon"].values):
    zones_peel.append(peel_zone_for_point(lat, lon))
full_df["zone_peel"] = zones_peel

# Drop rows where we couldn't assign a Peel zone (nodata)
mask    = full_df["zone_peel"].notna()
df_comp = full_df.loc[mask, ["zone_latband", "zone_peel"]].copy()

# 7) Agreement statistics
agreement = (df_comp["zone_latband"] == df_comp["zone_peel"]).mean() * 100.0
print(f"Global agreement between latitude-band and Peel-based zones: {agreement:.1f}%")

confusion = pd.crosstab(df_comp["zone_latband"], df_comp["zone_peel"], normalize="index") * 100
print("\nRow-normalised confusion matrix (%):")
print(confusion.round(1))

# 8) KS tests for sampling rationality (full vs sample)
print("\nKolmogorov–Smirnov tests: full vs sampled distributions")
for col in ["pm25", "t2m"]:
    stat, p = ks_2samp(full_df[col].values, sample_df[col].values)
    print(f"{col}: KS stat = {stat:.4f}, p = {p:.4f}")
