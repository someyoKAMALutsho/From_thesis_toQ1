"""
GPU-Accelerated Baseline Model Training with Köppen LOCO Cross-Validation
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (v3 — PLOS ONE Major Revision):
===============================================
- CRITICAL FIX: LightGBM now loads zone-specific hyperparameters from
  best_params_per_zone_revised.csv (output of 14_tuning_lightgbm.py v3).
  Previous version used a single hardcoded parameter set for all zones,
  which (a) ignored tuning output entirely and (b) used parameters tuned
  on Subtropical applied to all zones — data leakage.
- Paths updated from F: drive to D: drive (active working copy).
- Output files use _revised suffix to preserve original results.
- XGBoost, RF, LinearRegression, GAM parameters unchanged from v2.

CODE SOURCES & ATTRIBUTION:
===========================
1. XGBoost GPU Training:
   SOURCE: https://xgboost.readthedocs.io/en/stable/gpu/index.html
   REFERENCE: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting
   System. KDD'16.
   MODIFICATION: device='cuda' for NVIDIA RTX 3060. Parameters unchanged
   from v2 — XGBoost is not the primary model under revision.

2. LightGBM GPU Training + Zone-Specific Parameters:
   SOURCE: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
   REFERENCE: Ke et al. (2017). LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree. NeurIPS 2017.
   MODIFICATION: Parameters now loaded per-zone from CSV instead of
   hardcoded. device='cuda' retained for GPU acceleration.

3. Random Forest:
   SOURCE: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
   REFERENCE: Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
   MODIFICATION: None from v2.

4. Linear Regression:
   SOURCE: https://scikit-learn.org/stable/modules/linear_model.html
   MODIFICATION: None from v2.

5. Köppen LOCO Cross-Validation:
   SOURCE: Roberts et al. (2017). Cross-validation strategies for data with
   temporal, spatial, hierarchical, or phylogenetic structure.
   Ecography, 40(8), 913-929.
   MODIFICATION: None from v2. Imputer still fit on training zones only.

6. Loading tuned parameters per fold:
   SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
   MODIFICATION: Parameters loaded from CSV row matching held-out zone,
   then cast to correct Python types before passing to LGBMRegressor.

HARDWARE: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

try:
    from pygam import LinearGAM
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("WARNING: pygam not available, GAM will be skipped")

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE        = DATA_DIR / "pm25_merra2_meteorology_final.parquet"
TUNED_PARAMS_FILE = RESULTS_DIR / "best_params_per_zone_revised.csv"
XGB_TUNED_PARAMS_FILE = RESULTS_DIR / "best_params_xgb_per_zone_revised.csv"

# FEATURE SET LOCKED — do not change without Perplexity approval.
# 8-feature set empirically optimal for LOCO generalisation.
# 10-feature set tested (added blh, month_sin, month_cos, year_norm):
# mean R² dropped 0.367→0.333, Tropical R² dropped 0.142→0.100. Rejected.
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'
RANDOM_STATE = 42

print("=" * 70)
print("GPU-ACCELERATED BASELINE MODEL TRAINING (v3)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print("Fix: LightGBM now uses zone-specific tuned params from Script 14")
print("=" * 70)

# =============================================================================
# LOAD ZONE-SPECIFIC LIGHTGBM PARAMETERS
# SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# MODIFICATION: Loads tuning output from 14_tuning_lightgbm.py v3.
#   Each LOCO fold uses parameters tuned exclusively on its training zones.
# =============================================================================
print("\nLoading zone-specific LightGBM parameters...")
if not TUNED_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"Tuned params file not found: {TUNED_PARAMS_FILE}\n"
        "Run 14_tuning_lightgbm.py first."
    )

params_df = pd.read_csv(TUNED_PARAMS_FILE)
# Build dict: zone → parameter dict
# Column names in CSV are prefixed with 'param_' from script 14
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

print(f"  Loaded parameters for {len(ZONE_PARAMS)} zones:")
for zone, params in ZONE_PARAMS.items():
    print(f"  [{zone}] n_est={params['n_estimators']}  "
          f"depth={params['max_depth']}  lr={params['learning_rate']}  "
          f"leaves={params['num_leaves']}")

# =============================================================================
# GPU AVAILABILITY CHECK
# =============================================================================
print("\nTesting GPU availability...")
test_X = np.random.rand(100, len(FEATURE_COLS))
test_y = np.random.rand(100)

try:
    test_model = xgb.XGBRegressor(device='cuda', n_estimators=5, verbosity=0)
    test_model.fit(test_X, test_y)
    print("  XGBoost GPU (RTX 3060): Working")
    XGB_GPU = True
except Exception as e:
    print(f"  XGBoost GPU failed: {e} — falling back to CPU")
    XGB_GPU = False

try:
    test_lgb = lgb.LGBMRegressor(device='cuda', n_estimators=5, verbose=-1)
    test_lgb.fit(test_X, test_y)
    print("  LightGBM GPU (RTX 3060): Working")
    LGB_GPU = True
except Exception as e:
    print(f"  LightGBM GPU failed: {e} — falling back to CPU")
    LGB_GPU = False

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
zones   = sorted(df_full['koppen_zone'].unique())
print(f"  Total samples : {len(df_full):,}")
print(f"  Climate zones : {zones}")

# Verify all zones have tuned parameters
missing_zones = [z for z in zones if z not in ZONE_PARAMS]
if missing_zones:
    raise ValueError(
        f"No tuned parameters found for zones: {missing_zones}\n"
        "Re-run 14_tuning_lightgbm.py to regenerate parameters."
    )
print("  All zones have tuned parameters.")

# =============================================================================
# RESULTS STORAGE
# =============================================================================
results = {
    'zone': [], 'model': [], 'train_samples': [], 'test_samples': [],
    'mae': [], 'rmse': [], 'r2': [], 'bias': [],
    'explained_variance': [], 'train_time_sec': []
}

def record(zone, model_name, n_train, n_test, y_true, y_pred, elapsed):
    results['zone'].append(zone)
    results['model'].append(model_name)
    results['train_samples'].append(n_train)
    results['test_samples'].append(n_test)
    results['mae'].append(mean_absolute_error(y_true, y_pred))
    results['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
    results['r2'].append(r2_score(y_true, y_pred))
    results['bias'].append(float(np.mean(y_pred - y_true)))
    results['explained_variance'].append(
        float(1 - np.var(y_true - y_pred) / np.var(y_true))
    )
    results['train_time_sec'].append(elapsed)

# =============================================================================
# LOCO CROSS-VALIDATION — MAIN LOOP
# SOURCE: Roberts et al. (2017). Ecography, 40(8), 913-929.
# MODIFICATION: LightGBM uses zone-specific parameters loaded above.
#   All other models unchanged from v2.
# =============================================================================
print("\n" + "=" * 70)
print("LOCO CROSS-VALIDATION")
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

    # Imputation — fit on train only
    # SOURCE: https://scikit-learn.org/stable/modules/impute.html
    # MODIFICATION: Fresh imputer per fold, fit on training zones only.
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURE_COLS
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURE_COLS
    )

    # Scaling for linear models — fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ===== Model 1: XGBoost =====
    label = "GPU" if XGB_GPU else "CPU"
    print(f"  XGBoost ({label})...", end=" ", flush=True)
    t0 = time.time()
    # CRITICAL FIX (Fix 4): Load zone-specific tuned XGBoost parameters.
    # SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
    # REFERENCE: Script 15 (15_tuning_xgboost.py) output — nested LOCO-CV tuned params.
    # MODIFICATION: Replaces hardcoded fixed params with per-zone tuned params
    # from best_params_xgb_per_zone_revised.csv, identical loading pattern to
    # LightGBM zone params above. Ensures fair apples-to-apples comparison.
    if not hasattr(record, '_xgb_params_df'):
        if not XGB_TUNED_PARAMS_FILE.exists():
            raise FileNotFoundError(
                f"XGBoost tuned params not found: {XGB_TUNED_PARAMS_FILE}\n"
                "Run 15_tuning_xgboost.py first."
            )
        _xgb_params_raw = pd.read_csv(XGB_TUNED_PARAMS_FILE)
        record._xgb_params_df = {}
        for _, row in _xgb_params_raw.iterrows():
            record._xgb_params_df[row['zone']] = {
                'n_estimators'    : int(row['n_estimators']),
                'max_depth'       : int(row['max_depth']),
                'learning_rate'   : float(row['learning_rate']),
                'subsample'       : float(row['subsample']),
                'colsample_bytree': float(row['colsample_bytree']),
                'min_child_weight': int(row['min_child_weight']),
                'reg_alpha'       : float(row['reg_alpha']),
                'reg_lambda'      : float(row['reg_lambda']),
            }
    xgb_params = record._xgb_params_df[test_zone].copy()
    xgb_params['random_state'] = RANDOM_STATE
    xgb_params['verbosity']    = 0
    if XGB_GPU:
        xgb_params['device'] = 'cuda'
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    elapsed = time.time() - t0
    record(test_zone, f'XGBoost-{label}', n_train, n_test,
           y_test, y_pred_xgb, elapsed)
    print(f"R²={r2_score(y_test,y_pred_xgb):.4f}  "
          f"MAE={mean_absolute_error(y_test,y_pred_xgb):.3f}  ({elapsed:.1f}s)")

    # ===== Model 2: LightGBM — zone-specific tuned parameters =====
    # CRITICAL FIX: parameters loaded from best_params_per_zone_revised.csv
    # Each zone uses params tuned exclusively on its training zones (no leakage)
    label = "GPU" if LGB_GPU else "CPU"
    print(f"  LightGBM ({label}, zone-tuned params)...", end=" ", flush=True)
    t0 = time.time()
    lgb_params = ZONE_PARAMS[test_zone].copy()
    lgb_params['random_state'] = RANDOM_STATE
    lgb_params['verbose'] = -1
    if LGB_GPU:
        lgb_params['device'] = 'cuda'
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    elapsed = time.time() - t0
    record(test_zone, f'LightGBM-{label}', n_train, n_test,
           y_test, y_pred_lgb, elapsed)
    print(f"R²={r2_score(y_test,y_pred_lgb):.4f}  "
          f"MAE={mean_absolute_error(y_test,y_pred_lgb):.3f}  ({elapsed:.1f}s)")
    print(f"    Params used: {ZONE_PARAMS[test_zone]}")

    # ===== Model 3: Random Forest =====
    print(f"  Random Forest (CPU)...", end=" ", flush=True)
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    elapsed = time.time() - t0
    record(test_zone, 'Random Forest-CPU', n_train, n_test,
           y_test, y_pred_rf, elapsed)
    print(f"R²={r2_score(y_test,y_pred_rf):.4f}  "
          f"MAE={mean_absolute_error(y_test,y_pred_rf):.3f}  ({elapsed:.1f}s)")

    # ===== Model 4: Linear Regression =====
    print(f"  Linear Regression (CPU)...", end=" ", flush=True)
    t0 = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    elapsed = time.time() - t0
    record(test_zone, 'Linear Regression-CPU', n_train, n_test,
           y_test, y_pred_lr, elapsed)
    print(f"R²={r2_score(y_test,y_pred_lr):.4f}  "
          f"MAE={mean_absolute_error(y_test,y_pred_lr):.3f}  ({elapsed:.1f}s)")

    # ===== Model 5: GAM =====
    if HAS_GAM:
        print(f"  GAM (CPU)...", end=" ", flush=True)
        t0 = time.time()
        gam_model = LinearGAM()
        gam_model.fit(X_train_scaled, y_train)
        y_pred_gam = gam_model.predict(X_test_scaled)
        elapsed = time.time() - t0
        record(test_zone, 'GAM-CPU', n_train, n_test,
               y_test, y_pred_gam, elapsed)
        print(f"R²={r2_score(y_test,y_pred_gam):.4f}  "
              f"MAE={mean_absolute_error(y_test,y_pred_gam):.3f}  ({elapsed:.1f}s)")

# =============================================================================
# SAVE RESULTS
# _revised suffix preserves original files per project convention
# =============================================================================
df_results = pd.DataFrame(results)
results_csv = RESULTS_DIR / "loco_cv_results_revised.csv"
df_results.to_csv(results_csv, index=False)
print(f"\nSaved: {results_csv}")

print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY (Mean Across All Zones)")
print("=" * 70)
summary = df_results.groupby('model')[['mae', 'rmse', 'r2']].mean()
print(summary.round(4).to_string())

print("\nPerformance by Köppen Zone (mean across models):")
zone_summary = df_results.groupby('zone')[['mae', 'rmse', 'r2']].mean()
print(zone_summary.round(4).to_string())

# =============================================================================
# FIGURES
# Matplotlib source: https://matplotlib.org/stable/api/index.html
# MODIFICATION: Output filenames use _revised suffix. DPI=300 retained.
# =============================================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pivot_r2 = df_results.pivot_table(
    index='zone', columns='model', values='r2', aggfunc='mean'
)
pivot_r2.plot(kind='bar', ax=axes[0])
axes[0].set_title('R² Score by Köppen Zone (Revised)', fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_xlabel('Köppen Climate Zone')
axes[0].legend(title='Model', fontsize=7)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

pivot_mae = df_results.pivot_table(
    index='zone', columns='model', values='mae', aggfunc='mean'
)
pivot_mae.plot(kind='bar', ax=axes[1])
axes[1].set_title('MAE by Köppen Zone (Revised)', fontweight='bold')
axes[1].set_ylabel('MAE (µg m⁻³)')
axes[1].set_xlabel('Köppen Climate Zone')
axes[1].legend(title='Model', fontsize=7)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "performance_by_zone_revised.png",
            dpi=300, bbox_inches='tight')
print("  Saved: performance_by_zone_revised.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
model_r2 = df_results.groupby('model')['r2'].mean().sort_values(ascending=False)
model_r2.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Mean R² Score by Model — LOCO CV (Revised)', fontweight='bold')
ax.set_xlabel('R² Score')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "r2_by_model_revised.png",
            dpi=300, bbox_inches='tight')
print("  Saved: r2_by_model_revised.png")
plt.close()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"\nOutputs saved to: {RESULTS_DIR}")
print("  loco_cv_results_revised.csv   ← Main results table for paper")
print("  performance_by_zone_revised.png")
print("  r2_by_model_revised.png")
print("\nNext: Run 17_ensemble_stacking.py with revised parameters")
print("=" * 70)
