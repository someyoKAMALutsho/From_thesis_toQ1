"""
Latitude/Longitude Ablation Study
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

PURPOSE (Reviewer Task B):
==========================
Reviewer requested evidence that geographic coordinates (lat, lon) provide
genuine predictive signal rather than acting as proxies for training data
density or spatial autocorrelation artefacts. This script quantifies the
performance penalty of removing lat and lon under the same LOCO-CV design
used in scripts 13-20, providing a direct empirical answer.

Three configurations tested under identical LOCO-CV:
  A) Full 8-feature set (canonical model — baseline)
  B) 6-feature set: meteorology + temporal only (lat + lon removed)
  C) 4-feature set: meteorology only (lat + lon + year + month removed)

Configuration C is included to isolate whether temporal features partially
substitute for geographic information when coordinates are absent.

SCRIPT DESIGN NOTES:
====================
- Drop-one (drop-group) retrain ablation: model retrained from scratch
  without the ablated features. This is methodologically stronger than
  permutation importance for quantifying substitutability, because
  permutation importance measures reliance on a feature in a trained model
  but cannot capture how the model would reorganise given the feature's
  absence. Rationale: Strobl et al. (2008).
- Zone-specific LightGBM parameters loaded from best_params_per_zone_revised.csv
  for all three configurations — identical to scripts 13/17/19/20.
- Results saved as lat_lon_ablation_results.csv.

CODE SOURCES & ATTRIBUTION:
===========================
1. Drop-one retrain ablation design:
   SOURCE: https://www.youtube.com/watch?v=CfHozJrnVLU (LightGBM drop-one
   ablation for feature evaluation — retrain and compare metric)
   REFERENCE: Strobl, C. et al. (2008). Conditional variable importance
   for random forests. BMC Bioinformatics, 9, 307.
   https://doi.org/10.1186/1471-2105-9-307
   MODIFICATION: Applied to LightGBM under LOCO-CV (not random forest under
   standard CV). Three configurations tested instead of single drop-one.

2. LOCO cross-validation design:
   SOURCE: https://scikit-learn.org/stable/modules/cross_validation.html
   REFERENCE: Roberts et al. (2017). Cross-validation strategies for data
   with temporal, spatial, hierarchical, or phylogenetic structure.
   Ecography, 40(8), 913-929. https://doi.org/10.1111/ecog.02881
   MODIFICATION: Outer fold = climate zone holdout. Imputer fit on training
   zones only per fold to prevent leakage.

3. LightGBM zone-specific parameters:
   SOURCE: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
   REFERENCE: Ke et al. (2017). LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree. NeurIPS 2017.
   MODIFICATION: Same parameter loading pattern as scripts 13/17/19/20
   for consistency.

4. Median imputation per fold:
   SOURCE: https://scikit-learn.org/stable/modules/impute.html
   MODIFICATION: Fit strictly on training zones, transform test zone
   separately — identical to all prior scripts.

HARDWARE: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
NOTE: LightGBM runs on CPU (pip build without CUDA).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# SOURCE: https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE        = DATA_DIR / "pm25_merra2_meteorology_final.parquet"
TUNED_PARAMS_FILE = RESULTS_DIR / "best_params_per_zone_revised.csv"
LOCO_RESULTS_FILE = RESULTS_DIR / "loco_cv_results_revised.csv"

RANDOM_STATE = 42
TARGET_COL   = 'pm25'

# Three ablation configurations
CONFIGS = {
    'A_full_8feat': {
        'features'   : ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10'],
        'description': 'Full model (canonical 8 features)',
        'label'      : 'Full (8 feat)',
    },
    'B_no_latlon': {
        'features'   : ['year', 'month', 't2m', 'ps', 'u10', 'v10'],
        'description': 'No geographic coordinates (lat + lon removed)',
        'label'      : 'No lat/lon (6 feat)',
    },
    'C_meteo_only': {
        'features'   : ['t2m', 'ps', 'u10', 'v10'],
        'description': 'Meteorology only (lat + lon + year + month removed)',
        'label'      : 'Meteo only (4 feat)',
    },
}

print("=" * 70)
print("LAT/LON ABLATION STUDY (Task B — Reviewer Response)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print("Configurations:")
for k, v in CONFIGS.items():
    print(f"  {k}: {v['description']}")
    print(f"       Features: {v['features']}")
print("=" * 70)

# =============================================================================
# LOAD ZONE-SPECIFIC PARAMETERS
# SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# MODIFICATION: Same loading pattern as scripts 13/17/19/20.
# =============================================================================
print("\nLoading zone-specific LightGBM parameters...")
if not TUNED_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"Tuned params not found: {TUNED_PARAMS_FILE}\n"
        "Run 14_tuning_lightgbm.py first."
    )

params_df = pd.read_csv(TUNED_PARAMS_FILE)
ZONE_PARAMS = {}
for _, row in params_df.iterrows():
    zone = row['zone']
    ZONE_PARAMS[zone] = {
        'n_estimators'     : int(row['n_estimators']),
        'max_depth'        : int(row['max_depth']),
        'learning_rate'    : float(row['learning_rate']),
        'subsample'        : float(row['subsample']),
        'colsample_bytree' : float(row['colsample_bytree']),
        'num_leaves'       : int(row['num_leaves']),
        'min_child_samples': int(row['min_child_samples']),
        'random_state'     : RANDOM_STATE,
        'verbose'          : -1,
    }
print(f"  Loaded parameters for {len(ZONE_PARAMS)} zones.")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
zones   = sorted(df_full['koppen_zone'].unique())
print(f"  Total samples : {len(df_full):,}")
print(f"  Climate zones : {zones}")

# =============================================================================
# LOCO-CV ABLATION LOOP
# SOURCE: https://scikit-learn.org/stable/modules/cross_validation.html
# REFERENCE: Roberts et al. (2017). Ecography, 40(8), 913-929.
# MODIFICATION: Three feature configurations run under identical LOCO design.
# Imputer re-fit per configuration per fold to prevent any cross-config leakage.
# =============================================================================
all_results = []

for config_id, config in CONFIGS.items():
    feat_cols = config['features']
    print(f"\n{'=' * 70}")
    print(f"CONFIG {config_id}: {config['description']}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"{'=' * 70}")

    zone_r2s, zone_maes, zone_rmses = [], [], []

    for test_zone in zones:
        train_data = df_full[df_full['koppen_zone'] != test_zone].copy()
        test_data  = df_full[df_full['koppen_zone'] == test_zone].copy()

        X_train = train_data[feat_cols].copy()
        y_train = train_data[TARGET_COL].copy()
        X_test  = test_data[feat_cols].copy()
        y_test  = test_data[TARGET_COL].copy()

        # Imputation — fit on training zones only
        # SOURCE: https://scikit-learn.org/stable/modules/impute.html
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train), columns=feat_cols
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test), columns=feat_cols
        )

        # Train with zone-specific params — identical to script 13
        model = lgb.LGBMRegressor(**ZONE_PARAMS[test_zone])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        zone_r2s.append(r2)
        zone_maes.append(mae)
        zone_rmses.append(rmse)

        print(f"  {test_zone:<14}: R²={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

        all_results.append({
            'config'     : config_id,
            'label'      : config['label'],
            'zone'       : test_zone,
            'n_features' : len(feat_cols),
            'features'   : ', '.join(feat_cols),
            'r2'         : r2,
            'mae'        : mae,
            'rmse'       : rmse,
        })

    mean_r2   = np.mean(zone_r2s)
    mean_mae  = np.mean(zone_maes)
    mean_rmse = np.mean(zone_rmses)
    print(f"  {'MEAN':<14}: R²={mean_r2:.4f}  MAE={mean_mae:.3f}  RMSE={mean_rmse:.3f}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
df_results = pd.DataFrame(all_results)

print("\n" + "=" * 70)
print("ABLATION SUMMARY — MEAN ACROSS ALL ZONES")
print("=" * 70)

summary = df_results.groupby(
    ['config', 'label', 'n_features'], sort=False
)[['r2', 'mae', 'rmse']].mean().reset_index()

# Load canonical LightGBM R² from script 13 for cross-check
try:
    loco_df      = pd.read_csv(LOCO_RESULTS_FILE)
    lgb_loco_r2  = float(
        loco_df[loco_df['model'].str.startswith('LightGBM')]['r2'].mean()
    )
    print(f"  (Script 13 canonical LightGBM R² for reference: {lgb_loco_r2:.4f})")
except Exception:
    lgb_loco_r2 = None

print(f"\n  {'Config':<20} {'Features':>8} {'Mean R²':>9} "
      f"{'Mean MAE':>10} {'Mean RMSE':>11}")
print(f"  {'─'*62}")

full_r2 = None
for _, row in summary.iterrows():
    marker = " ← canonical" if row['config'] == 'A_full_8feat' else ""
    print(f"  {row['label']:<20} {int(row['n_features']):>8} "
          f"{row['r2']:>9.4f} {row['mae']:>10.3f} "
          f"{row['rmse']:>11.3f}{marker}")
    if row['config'] == 'A_full_8feat':
        full_r2 = row['r2']

# R² penalty table
print(f"\n  {'─'*62}")
print(f"  R² PENALTY FROM REMOVING GEOGRAPHIC FEATURES")
print(f"  {'─'*62}")
for _, row in summary.iterrows():
    if row['config'] == 'A_full_8feat':
        continue
    penalty    = row['r2'] - full_r2
    penalty_pct = penalty / abs(full_r2) * 100 if full_r2 else np.nan
    print(f"  {row['label']:<20}: ΔR² = {penalty:+.4f} ({penalty_pct:+.1f}%)")

# Per-zone breakdown
print(f"\n  {'─'*62}")
print(f"  PER-ZONE R² BY CONFIGURATION")
print(f"  {'─'*62}")
pivot = df_results.pivot_table(
    index='zone', columns='config', values='r2'
).round(4)
pivot['penalty_no_latlon'] = (
    pivot['B_no_latlon'] - pivot['A_full_8feat']
).round(4)
print(pivot.to_string())

# =============================================================================
# SAVE
# =============================================================================
out_path = RESULTS_DIR / "lat_lon_ablation_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

summary_path = RESULTS_DIR / "lat_lon_ablation_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"  Saved: {summary_path}")

# =============================================================================
# INTERPRETATION FOR PAPER
# =============================================================================
no_latlon_r2 = summary[summary['config'] == 'B_no_latlon']['r2'].values[0]
meteo_r2     = summary[summary['config'] == 'C_meteo_only']['r2'].values[0]
penalty_latlon = no_latlon_r2 - full_r2
penalty_pct    = penalty_latlon / abs(full_r2) * 100

print("\n" + "=" * 70)
print("INTERPRETATION FOR REVIEWER RESPONSE")
print("=" * 70)
print(f"  Removing lat + lon:   ΔR² = {penalty_latlon:+.4f} "
      f"({penalty_pct:+.1f}%)")
print(f"  Meteorology only:     ΔR² = {meteo_r2 - full_r2:+.4f} "
      f"({(meteo_r2 - full_r2)/abs(full_r2)*100:+.1f}%)")

if penalty_pct < -15:
    print("\n  Verdict: Geographic coordinates provide SUBSTANTIAL independent")
    print("  predictive signal. Removing lat/lon causes meaningful R² degradation.")
    print("  This confirms geographic features are not redundant proxies —")
    print("  they encode persistent PM2.5 structural gradients not captured")
    print("  by meteorology alone.")
elif penalty_pct < -5:
    print("\n  Verdict: Geographic coordinates provide MODERATE independent signal.")
else:
    print("\n  Verdict: Geographic coordinates provide limited independent signal")
    print("  beyond meteorology. Review SHAP analysis for further context.")

print("\n" + "=" * 70)
print("ABLATION COMPLETE")
print("=" * 70)
print(f"\nOutputs in: {RESULTS_DIR}")
print("  lat_lon_ablation_results.csv")
print("  lat_lon_ablation_summary.csv")
print("\nNext: Run spatial_baseline.py (Task C — Kriging/IDW comparison)")
print("=" * 70)
