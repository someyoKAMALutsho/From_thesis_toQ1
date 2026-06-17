"""
Ensemble Stacking: Meta-Learner Approach (LightGBM + XGBoost + Random Forest)
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (v3 — PLOS ONE Major Revision):
===============================================
- CRITICAL FIX: LightGBM base model now loads zone-specific hyperparameters
  from best_params_per_zone_revised.csv (output of 14_tuning_lightgbm.py v3).
  Previous version used a single hardcoded LightGBM parameter set for all
  zones — inconsistent with the nested CV tuning approach and constitutes
  the same leakage fixed in script 13.
- CRITICAL FIX: Baseline R² comparison now loaded dynamically from
  loco_cv_results_revised.csv instead of hardcoded scalar (0.341).
  Hardcoded baselines become stale when upstream scripts are rerun.
- Output file renamed stacking_results_revised.csv to preserve originals.
- Paths already dynamic from path-fix pass (no change needed).

CODE SOURCES & ATTRIBUTION:
===========================
1. Stacking Generalization (StackingRegressor):
   SOURCE: https://scikit-learn.org/stable/modules/ensemble.html#stacking
   REFERENCE: Wolpert, D.H. (1992). Stacked Generalization.
   Neural Networks, 5(2), 241-259.
   MODIFICATION: cv=5 inner folds operate strictly within training zones
   (training data only passed to fit()). Test zone never seen during
   meta-feature generation.

2. LightGBM base model — zone-specific parameters:
   SOURCE: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
   REFERENCE: Ke et al. (2017). LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree. NeurIPS 2017.
   MODIFICATION: Parameters loaded per zone from CSV instead of hardcoded.
   Ensures stacking ensemble uses same tuned parameters as standalone
   LightGBM in script 13 — comparison remains valid.

3. XGBoost base model:
   SOURCE: https://xgboost.readthedocs.io/en/stable/python/python_api.html
   REFERENCE: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting
   System. KDD'16.
   MODIFICATION: Parameters unchanged from v2. XGBoost used as fixed
   baseline comparator — not zone-tuned.

4. Random Forest base model:
   SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
   REFERENCE: Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
   MODIFICATION: Parameters unchanged from v2.

5. Meta-learner (Linear Regression):
   SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
   REFERENCE: Zhou (2012). Ensemble Methods: Foundations and Algorithms.
   CRC Press, Chapter 5.
   MODIFICATION: None. Linear meta-learner intentionally simple to prevent
   overfitting of the stacking layer.

6. Dynamic baseline comparison from CSV:
   SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
   MODIFICATION: Replaces hardcoded baseline scalar. LightGBM mean R²
   computed from loco_cv_results_revised.csv at runtime.

HARDWARE: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
NOTE: LightGBM runs on CPU (pip build without CUDA). XGBoost runs on GPU.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — Dynamic path resolution
# SOURCE: https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
# MODIFICATION: Replaces hardcoded paths. Resolves relative to script location.
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE        = DATA_DIR / "pm25_merra2_meteorology_final.parquet"
TUNED_PARAMS_FILE = RESULTS_DIR / "best_params_per_zone_revised.csv"
BASELINE_FILE     = RESULTS_DIR / "loco_cv_results_revised.csv"

# FEATURE SET LOCKED — 8 features, empirically optimal for LOCO generalisation.
# 10-feature set tested, mean R² dropped 0.367→0.333. Rejected.
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'
RANDOM_STATE = 42

print("=" * 70)
print("ENSEMBLE STACKING: META-LEARNER APPROACH (v3)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print("Fix: LightGBM base model uses zone-specific tuned params")
print("Fix: Baseline comparison loaded dynamically from CSV")
print("=" * 70)

# =============================================================================
# LOAD ZONE-SPECIFIC LIGHTGBM PARAMETERS
# SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# MODIFICATION: Same loading logic as script 13 — ensures consistency.
# =============================================================================
print("\nLoading zone-specific LightGBM parameters...")
if not TUNED_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"Tuned params file not found: {TUNED_PARAMS_FILE}\n"
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
    }
print(f"  Loaded parameters for {len(ZONE_PARAMS)} zones.")

# =============================================================================
# LOAD BASELINE FOR COMPARISON
# MODIFICATION: Dynamic load prevents stale hardcoded values after reruns.
# =============================================================================
print("Loading baseline results for comparison...")
if not BASELINE_FILE.exists():
    raise FileNotFoundError(
        f"Baseline results not found: {BASELINE_FILE}\n"
        "Run 13_baseline_training_GPU.py first."
    )

baseline_df = pd.read_csv(BASELINE_FILE)
lgb_baseline = baseline_df[
    baseline_df['model'].str.startswith('LightGBM')
].groupby('zone')['r2'].mean()
lgb_mean_r2 = lgb_baseline.mean()
print(f"  LightGBM baseline mean R²: {lgb_mean_r2:.4f}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
zones   = sorted(df_full['koppen_zone'].unique())
print(f"  Total samples : {len(df_full):,}")
print(f"  Climate zones : {zones}")

missing_zones = [z for z in zones if z not in ZONE_PARAMS]
if missing_zones:
    raise ValueError(f"No tuned parameters for zones: {missing_zones}")

# =============================================================================
# LOCO CROSS-VALIDATION WITH STACKING
# SOURCE: https://scikit-learn.org/stable/modules/ensemble.html#stacking
# REFERENCE: Wolpert (1992). Neural Networks, 5(2), 241-259.
# MODIFICATION: LightGBM parameters vary per fold. cv=5 meta-feature
# generation operates strictly within training zones only.
# =============================================================================
results = {
    'zone': [], 'model': [], 'mae': [], 'rmse': [], 'r2': [],
    'bias': [], 'train_time_sec': []
}

print("\n" + "=" * 70)
print("LOCO CROSS-VALIDATION WITH STACKING")
print("=" * 70)

for test_zone in zones:
    print(f"\n{'─' * 60}")
    print(f"Held-out zone: {test_zone}")
    print(f"{'─' * 60}")

    train_data = df_full[df_full['koppen_zone'] != test_zone].copy()
    test_data  = df_full[df_full['koppen_zone'] == test_zone].copy()
    n_train, n_test = len(train_data), len(test_data)
    print(f"  Train: {n_train:,}  |  Test: {n_test:,}")

    X_train = train_data[FEATURE_COLS].copy()
    y_train = train_data[TARGET_COL].copy()
    X_test  = test_data[FEATURE_COLS].copy()
    y_test  = test_data[TARGET_COL].copy()

    # Imputation — fit on training zones only
    # SOURCE: https://scikit-learn.org/stable/modules/impute.html
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURE_COLS
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURE_COLS
    )

    # Zone-specific LightGBM parameters for this fold
    zone_lgb_params = ZONE_PARAMS[test_zone].copy()
    zone_lgb_params['random_state'] = RANDOM_STATE
    zone_lgb_params['verbose'] = -1

    # Base models — LightGBM uses zone-tuned params, others fixed
    base_models = [
        ('lgb', lgb.LGBMRegressor(**zone_lgb_params)),
        ('xgb', xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=15,
            random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]

    # Meta-learner — intentionally simple (linear) to prevent overfitting
    meta_model = LinearRegression()

    print(f"  Building stacking ensemble...", end=" ", flush=True)
    t0 = time.time()

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    elapsed = time.time() - t0

    mae  = mean_absolute_error(y_test, y_pred_stack)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    r2   = r2_score(y_test, y_pred_stack)
    bias = float(np.mean(y_pred_stack - y_test))

    print(f"done in {elapsed:.1f}s")
    print(f"  R²={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}  Bias={bias:+.3f}")

    # Compare to standalone LightGBM for this zone
    lgb_zone_r2 = lgb_baseline.get(test_zone, np.nan)
    print(f"  vs LightGBM alone: R²={lgb_zone_r2:.4f}  "
          f"Gain: {r2 - lgb_zone_r2:+.4f}")

    results['zone'].append(test_zone)
    results['model'].append('Stacking (LGB+XGB+RF)')
    results['mae'].append(mae)
    results['rmse'].append(rmse)
    results['r2'].append(r2)
    results['bias'].append(bias)
    results['train_time_sec'].append(elapsed)

# =============================================================================
# SAVE AND SUMMARISE
# =============================================================================
df_stacking = pd.DataFrame(results)
out_path = RESULTS_DIR / "stacking_results_revised.csv"
df_stacking.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

stacking_mean_r2  = df_stacking['r2'].mean()
stacking_mean_mae = df_stacking['mae'].mean()
improvement       = stacking_mean_r2 - lgb_mean_r2

print("\n" + "=" * 70)
print("STACKING RESULTS SUMMARY")
print("=" * 70)
print(f"{'Zone':<14} {'Stack R²':>9} {'LGB R²':>8} {'Gain':>7} {'MAE':>8}")
print("─" * 50)
for zone in zones:
    s_r2 = df_stacking[df_stacking['zone'] == zone]['r2'].values[0]
    s_mae = df_stacking[df_stacking['zone'] == zone]['mae'].values[0]
    l_r2 = lgb_baseline.get(zone, np.nan)
    print(f"{zone:<14} {s_r2:>9.4f} {l_r2:>8.4f} "
          f"{s_r2 - l_r2:>+7.4f} {s_mae:>8.3f}")
print("─" * 50)
print(f"{'MEAN':<14} {stacking_mean_r2:>9.4f} {lgb_mean_r2:>8.4f} "
      f"{improvement:>+7.4f} {stacking_mean_mae:>8.3f}")

print("\n" + "=" * 70)
print("COMPARISON: Stacking vs Tuned LightGBM")
print("=" * 70)
print(f"Baseline LightGBM mean R² : {lgb_mean_r2:.4f}")
print(f"Stacking mean R²          : {stacking_mean_r2:.4f}")
print(f"Improvement               : {improvement:+.4f} "
      f"({improvement/lgb_mean_r2*100:+.1f}%)")

if improvement > 0.01:
    print("\nVerdict: Stacking provides meaningful improvement over LightGBM alone.")
    print("         Report stacking as primary model in manuscript.")
elif improvement > 0:
    print("\nVerdict: Marginal improvement — report both, use LightGBM as primary.")
else:
    print("\nVerdict: Stacking does not improve over tuned LightGBM.")
    print("         LightGBM remains primary model. Report stacking as ablation.")

print("\n" + "=" * 70)
print("STACKING COMPLETE")
print("=" * 70)
print(f"\nNext: Run 19_shap_importance.py")
