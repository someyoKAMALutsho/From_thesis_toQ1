"""
Temporal Validation: Train–Test Chronological Split (2019–2021 → 2022–2023)
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (v3 — PLOS ONE Major Revision):
===============================================
- CRITICAL FIX (Fix 3): Replaced global zone-mean parameter averaging with
  PredefinedSplit inner-loop tuning. Inner train = 2019-2020, inner val =
  2021, test = 2022-2023. Tuning strictly within training years — test years
  never seen during hyperparameter search. Eliminates heuristic averaging
  and closes the temporal leakage concern. Reference: Bergmeir & Benitez
  (2012). Information Sciences, 191, 192-213.
- CRITICAL FIX: LOCO comparison now reads loco_cv_results_revised.csv
  (output of script 13 v3). Previous version read loco_cv_results_gpu.csv
  which no longer exists after path-fix pass.
- Added per-year breakdown of test performance (2022 vs 2023 separately)
  to detect temporal drift within the test period.
- Added per-zone breakdown of temporal validation performance to match
  the LOCO zone-level reporting format.
- Output renamed temporal_vs_loco_validation_revised.csv.

CODE SOURCES & ATTRIBUTION:
===========================
1. Chronological train/test split rationale:
   SOURCE: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
   REFERENCE: Bergmeir & Benitez (2012). On the use of cross-validation for
   time series predictor evaluation. Information Sciences, 191, 192-213.
   MODIFICATION: Single split at 2021/2022 boundary rather than rolling
   window — justified by the paper's focus on spatial generalisation.
   Temporal split is a secondary validation check, not the primary design.

2. LightGBM parameter averaging across zones:
   SOURCE: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
   MODIFICATION: Integer params (n_estimators, max_depth, num_leaves,
   min_child_samples) rounded to nearest integer after averaging.
   Float params (learning_rate, subsample, colsample_bytree) averaged directly.

3. Median imputation — fit on training years only:
   SOURCE: https://scikit-learn.org/stable/modules/impute.html
   MODIFICATION: Identical to scripts 13/17/19 — imputer fit on train split
   only, transform applied to test split separately.

4. Per-zone breakdown:
   SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
   MODIFICATION: Applied to temporal test set to match LOCO reporting format.

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

# FEATURE SET LOCKED — 8 features empirically optimal.
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'
RANDOM_STATE = 42

TRAIN_YEARS = [2019, 2020, 2021]
TEST_YEARS  = [2022, 2023]

print("=" * 70)
print("TEMPORAL VALIDATION: CHRONOLOGICAL SPLIT (v3)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Train: {TRAIN_YEARS}  →  Test: {TEST_YEARS}")
print("Fix 3: PredefinedSplit temporal tuning (2019-2020 inner train, 2021 val)")
print("Fix: Reads loco_cv_results_revised.csv for comparison")
print("=" * 70)



# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
print(f"  Total samples : {len(df_full):,}")
print(f"  Year range    : {df_full['year'].min()}–{df_full['year'].max()}")
print(f"  Zones         : {sorted(df_full['koppen_zone'].unique())}")

train_data = df_full[df_full['year'].isin(TRAIN_YEARS)].copy()
test_data  = df_full[df_full['year'].isin(TEST_YEARS)].copy()

print(f"\n  Train {TRAIN_YEARS}: {len(train_data):,} samples")
print(f"  Test  {TEST_YEARS} : {len(test_data):,} samples")

X_train = train_data[FEATURE_COLS].copy()
y_train = train_data[TARGET_COL].copy()
X_test  = test_data[FEATURE_COLS].copy()
y_test  = test_data[TARGET_COL].copy()

# =============================================================================
# IMPUTATION — fit on training years only
# SOURCE: https://scikit-learn.org/stable/modules/impute.html
# MODIFICATION: Identical pattern to scripts 13/17/19 — no leakage.
# =============================================================================
print("\nImputing missing values (median, fit on train only)...", end=" ")
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLS)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=FEATURE_COLS)
print("done")

# =============================================================================
# FIX 3: TEMPORAL HYPERPARAMETER TUNING via PredefinedSplit
# Replaces global zone-mean averaging with a principled temporal inner loop.
# Inner train = 2019-2020, inner val = 2021, test = 2022-2023.
# This ensures tuning never sees the test years (2022-2023).
#
# SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html
# REFERENCE: Bergmeir & Benitez (2012). On the use of cross-validation for
#   time series predictor evaluation. Information Sciences, 191, 192-213.
# MODIFICATION: PredefinedSplit used instead of TimeSeriesSplit because the
#   validation boundary is fixed at 2021 (not rolling). Inner train =
#   2019-2020 (fold_id=-1, never held out), inner val = 2021 (fold_id=0).
#   This is the methodologically correct approach for a single temporal split.
#
# SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# REFERENCE: Bergstra & Bengio (2012). JMLR, 13, 281-305.
# MODIFICATION: n_iter=30, same as Scripts 14 and 15, for consistency.
# =============================================================================
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

print("\nFix 3: Temporal hyperparameter tuning via PredefinedSplit...")
print("  Inner train: 2019-2020  |  Inner val: 2021  |  Test: 2022-2023")

INNER_TRAIN_YEARS = [2019, 2020]
INNER_VAL_YEARS   = [2021]

# Build inner split indices from training data (2019-2021)
# fold_id = -1 → always in inner train (2019-2020)
# fold_id =  0 → held out as inner val (2021)
# SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html
fold_ids = np.where(
    train_data['year'].isin(INNER_VAL_YEARS), 0, -1
)
ps = PredefinedSplit(test_fold=fold_ids)
print(f"  Inner train size: {(fold_ids == -1).sum():,}  "
      f"|  Inner val size: {(fold_ids == 0).sum():,}")

# Search space — identical to Scripts 14 and 15 for consistency
# SOURCE: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# REFERENCE: Ke et al. (2017). LightGBM: A Highly Efficient Gradient
#   Boosting Decision Tree. NeurIPS 2017.
param_dist_temporal = {
    'n_estimators'     : [100, 200, 300, 400, 500],
    'max_depth'        : [4, 5, 6, 7, 8, 10],
    'learning_rate'    : [0.01, 0.02, 0.05, 0.1, 0.15],
    'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
    'num_leaves'       : [20, 31, 50, 70],
    'min_child_samples': [10, 20, 30, 50],
}

lgb_base = lgb.LGBMRegressor(
    verbose=-1, random_state=RANDOM_STATE, n_jobs=-1
)

inner_search = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=param_dist_temporal,
    n_iter=30,
    cv=ps,
    scoring='r2',
    n_jobs=-1,
    verbose=0,
    random_state=RANDOM_STATE,
    return_train_score=True
)

print("  Running inner search (30 iter × PredefinedSplit)...",
      end=" ", flush=True)
inner_search.fit(X_train, y_train)
print("done")

GLOBAL_LGB_PARAMS = inner_search.best_params_
GLOBAL_LGB_PARAMS['random_state'] = RANDOM_STATE
GLOBAL_LGB_PARAMS['verbose']      = -1

print(f"  Best inner val R²: {inner_search.best_score_:.4f}")
print("  Tuned parameters:")
for k, v in GLOBAL_LGB_PARAMS.items():
    if k not in ('random_state', 'verbose'):
        print(f"    {k:<20}: {v}")

# Retrain on full training set (2019-2021) with tuned parameters
# SOURCE: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
# MODIFICATION: Final model trained on all TRAIN_YEARS (2019-2021) after
#   tuning. Test set (2022-2023) never seen during tuning — no leakage.
print(f"\nRetraining on full {TRAIN_YEARS} with tuned parameters...",
      end=" ", flush=True)
lgb_model = lgb.LGBMRegressor(**GLOBAL_LGB_PARAMS)
lgb_model.fit(X_train, y_train)
print("done")

y_pred = lgb_model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
bias = float(np.mean(y_pred - y_test))

print(f"\n{'─' * 50}")
print(f"OVERALL TEMPORAL TEST ({TEST_YEARS})")
print(f"{'─' * 50}")
print(f"  R²  : {r2:.4f}")
print(f"  MAE : {mae:.3f} µg m⁻³")
print(f"  RMSE: {rmse:.3f}")
print(f"  Bias: {bias:+.3f} µg m⁻³")

# =============================================================================
# PER-YEAR BREAKDOWN — detect temporal drift within test period
# SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# MODIFICATION: Splits 2022 and 2023 results to check for year-to-year drift.
# =============================================================================
print(f"\n{'─' * 50}")
print("PER-YEAR BREAKDOWN (2022 vs 2023)")
print(f"{'─' * 50}")

test_data_eval = test_data.copy()
test_data_eval['y_pred'] = y_pred

per_year_rows = []
for yr in TEST_YEARS:
    mask  = test_data_eval['year'] == yr
    y_t   = test_data_eval.loc[mask, TARGET_COL]
    y_p   = test_data_eval.loc[mask, 'y_pred']
    yr_r2   = r2_score(y_t, y_p)
    yr_mae  = mean_absolute_error(y_t, y_p)
    yr_rmse = np.sqrt(mean_squared_error(y_t, y_p))
    yr_bias = float(np.mean(y_p - y_t))
    print(f"  {yr}: R²={yr_r2:.4f}  MAE={yr_mae:.3f}  "
          f"RMSE={yr_rmse:.3f}  Bias={yr_bias:+.3f}")
    per_year_rows.append({
        'split': f'Temporal_{yr}',
        'r2': yr_r2, 'mae': yr_mae,
        'rmse': yr_rmse, 'bias': yr_bias
    })

# =============================================================================
# PER-ZONE BREAKDOWN — match LOCO zone-level reporting
# =============================================================================
print(f"\n{'─' * 50}")
print("PER-ZONE BREAKDOWN (temporal test set)")
print(f"{'─' * 50}")

per_zone_rows = []
for zone in sorted(test_data_eval['koppen_zone'].unique()):
    mask  = test_data_eval['koppen_zone'] == zone
    y_t   = test_data_eval.loc[mask, TARGET_COL]
    y_p   = test_data_eval.loc[mask, 'y_pred']
    z_r2   = r2_score(y_t, y_p)
    z_mae  = mean_absolute_error(y_t, y_p)
    z_rmse = np.sqrt(mean_squared_error(y_t, y_p))
    z_bias = float(np.mean(y_p - y_t))
    print(f"  {zone:<14}: R²={z_r2:.4f}  MAE={z_mae:.3f}  "
          f"RMSE={z_rmse:.3f}  Bias={z_bias:+.3f}")
    per_zone_rows.append({
        'split': f'Temporal_{zone}',
        'r2': z_r2, 'mae': z_mae,
        'rmse': z_rmse, 'bias': z_bias
    })

# =============================================================================
# LOAD LOCO METRICS FOR COMPARISON
# MODIFICATION: Reads loco_cv_results_revised.csv (script 13 v3 output).
# Previous version read loco_cv_results_gpu.csv — file no longer exists.
# =============================================================================
print(f"\n{'─' * 50}")
print("COMPARISON: LOCO vs TEMPORAL")
print(f"{'─' * 50}")

if not LOCO_RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"LOCO results not found: {LOCO_RESULTS_FILE}\n"
        "Run 13_baseline_training_GPU.py first."
    )

loco_df  = pd.read_csv(LOCO_RESULTS_FILE)
lgb_loco = loco_df[loco_df['model'].str.startswith('LightGBM')]
loco_r2   = float(lgb_loco['r2'].mean())
loco_mae  = float(lgb_loco['mae'].mean())
loco_rmse = float(lgb_loco['rmse'].mean())

print(f"\n  {'Metric':<8} {'LOCO':>10} {'Temporal':>10} {'Diff':>10}")
print(f"  {'─'*40}")
print(f"  {'R²':<8} {loco_r2:>10.4f} {r2:>10.4f} {r2-loco_r2:>+10.4f}")
print(f"  {'MAE':<8} {loco_mae:>10.3f} {mae:>10.3f} {mae-loco_mae:>+10.3f}")
print(f"  {'RMSE':<8} {loco_rmse:>10.3f} {rmse:>10.3f} {rmse-loco_rmse:>+10.3f}")

r2_diff_pct = (r2 - loco_r2) / abs(loco_r2) * 100 if loco_r2 != 0 else np.nan
print(f"\n  R² relative change: {r2_diff_pct:+.1f}%")

if r2_diff_pct > 10:
    print("\n  Interpretation: Temporal validation BETTER than LOCO.")
    print("  Model generalises more easily across time than across climate zones.")
    print("  Supports argument that geographic regime shift is the harder challenge.")
elif r2_diff_pct < -20:
    print("\n  Interpretation: Temporal validation WORSE than LOCO.")
    print("  Possible temporal covariate shift (emissions policy, data quality).")
    print("  Discuss in manuscript limitations.")
else:
    print("\n  Interpretation: Temporal and LOCO performance broadly consistent.")
    print("  Model is stable across both validation strategies.")

# =============================================================================
# SAVE RESULTS
# =============================================================================
summary_rows = [
    {'split': 'LOCO_mean',     'r2': loco_r2,  'mae': loco_mae,
     'rmse': loco_rmse, 'bias': np.nan},
    {'split': 'Temporal_all',  'r2': r2,        'mae': mae,
     'rmse': rmse,      'bias': bias},
] + per_year_rows + per_zone_rows

df_out = pd.DataFrame(summary_rows)
out_path = RESULTS_DIR / "temporal_vs_loco_validation_revised.csv"
df_out.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION COMPLETE (v3)")
print("=" * 70)
print(f"\nOutputs in: {RESULTS_DIR}")
print("  temporal_vs_loco_validation_revised.csv")
print("\nNext: Run lat_lon_ablation.py (Task B — reviewer request)")
print("=" * 70)
