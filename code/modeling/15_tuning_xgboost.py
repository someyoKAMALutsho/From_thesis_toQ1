"""
XGBoost Hyperparameter Tuning via Nested LOCO Cross-Validation
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (Fix 1 — PLOS ONE Major Revision):
==================================================
- NEW SCRIPT: XGBoost equivalent of 14_tuning_lightgbm.py.
  Previous Script 13 used fixed XGBoost parameters, making the model
  comparison unfair against zone-tuned LightGBM. This script implements
  identical nested LOCO-CV design so XGBoost is tuned on the same rigorous
  framework.
- GPU acceleration via tree_method='hist', device='cuda' with automatic
  CPU fallback if CUDA is unavailable.
- Output: best_params_xgb_per_zone_revised.csv — consumed by Fix 4
  (Script 13 update) and Fix 2 (Script 19 SHAP consistency update).

CODE SOURCES & ATTRIBUTION:
===========================
1. Nested Cross-Validation to prevent leakage:
   SOURCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
   REFERENCE: Cawley, G.C. & Talbot, N.L.C. (2010). On Over-fitting in Model
   Selection and Subsequent Selection Bias in Performance Evaluation.
   Journal of Machine Learning Research, 11, 2079-2107.
   MODIFICATION: Outer loop is LOCO zone split (leave-one-zone-out); inner
   loop is RandomizedSearchCV strictly on remaining training zones. The
   held-out zone is never seen during tuning. Identical outer structure
   to 14_tuning_lightgbm.py.

2. RandomizedSearchCV:
   SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
   REFERENCE: Bergstra & Bengio (2012). Random Search for Hyper-Parameter
   Optimization. Journal of Machine Learning Research, 13, 281-305.
   MODIFICATION: n_iter=30 per fold (same as LightGBM script). cv=3 inner
   splits retained for consistency across tuning scripts.

3. XGBoost Parameters and GPU usage:
   SOURCE: https://xgboost.readthedocs.io/en/stable/parameter.html
   SOURCE: https://xgboost.readthedocs.io/en/stable/gpu_support.html
   REFERENCE: Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree
   Boosting System. KDD 2016, pp. 785-794.
   MODIFICATION: tree_method='hist' with device='cuda' for RTX 3060 GPU.
   CPU fallback added via try/except to ensure script runs in any environment.
   n_jobs not passed when device='cuda' (XGBoost handles parallelism via GPU).

4. SimpleImputer fit-on-train-only pattern:
   SOURCE: https://scikit-learn.org/stable/modules/impute.html
   MODIFICATION: Imputer instantiated fresh per fold and fit only on training
   zones to prevent leakage of imputation statistics from held-out zone.
   Identical pattern to 14_tuning_lightgbm.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# Paths verified against D:\PM25_Satellite_Research folder inventory
# Identical path structure to 14_tuning_lightgbm.py
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"

# Feature set LOCKED — identical to all other modeling scripts.
# blh removed: 100% missing -> constant after imputation, zero signal.
# wind_speed removed: collinear with u10/v10 (reviewer multicollinearity concern).
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'

N_ITER_SEARCH = 30
N_INNER_CV    = 3
RANDOM_STATE  = 42

# =============================================================================
# GPU AVAILABILITY CHECK
# SOURCE: https://xgboost.readthedocs.io/en/stable/gpu_support.html
# MODIFICATION: Graceful CPU fallback so script runs on any machine.
# device='cuda' is the correct API for XGBoost >= 2.0 (replaces gpu_hist).
# =============================================================================
try:
    _probe = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
    _probe.fit(np.zeros((10, 2)), np.zeros(10))
    DEVICE     = 'cuda'
    TREE_METHOD = 'hist'
    print("GPU detected — running with device='cuda', tree_method='hist'")
except Exception:
    DEVICE      = 'cpu'
    TREE_METHOD = 'hist'
    print("GPU not available — falling back to CPU (tree_method='hist')")

print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING — NESTED LOCO CV (Fix 1)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print("Method  : Nested LOCO CV (outer=zone holdout, inner=RandomizedSearchCV)")
print("Fix     : Tuning strictly within training folds — identical to Script 14")
print(f"Features: {FEATURE_COLS}")
print(f"Device  : {DEVICE}  |  tree_method: {TREE_METHOD}")
print(f"Inner CV: {N_INNER_CV}-fold, {N_ITER_SEARCH} iterations per fold")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
zones   = sorted(df_full['koppen_zone'].unique())
print(f"  Total samples : {len(df_full):,}")
print(f"  Climate zones : {zones}")

# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# SOURCE: https://xgboost.readthedocs.io/en/stable/parameter.html
# REFERENCE: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System.
# MODIFICATION: Search space mirrors LightGBM Script 14 in spirit but uses
# XGBoost-native parameter names. num_leaves replaced by max_depth (XGBoost
# uses depth-wise growth, not leaf-wise). colsample_bytree kept equivalent.
# min_child_weight replaces min_child_samples (different but analogous).
# =============================================================================
param_dist = {
    'n_estimators'    : [100, 200, 300, 400, 500],
    'max_depth'       : [4, 5, 6, 7, 8, 10],
    'learning_rate'   : [0.01, 0.02, 0.05, 0.1, 0.15],
    'subsample'       : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 10, 20],
    'reg_alpha'       : [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda'      : [0.5, 1.0, 2.0, 5.0, 10.0],
}

BASELINE_DEFAULTS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 1, 'reg_alpha': 0, 'reg_lambda': 1.0,
}

total_combinations = np.prod([len(v) for v in param_dist.values()])
print(f"\nSearch space: {total_combinations:,} combinations")
print(f"Sampling    : {N_ITER_SEARCH} random per fold x {len(zones)} folds")

# =============================================================================
# NESTED LOCO CV — MAIN LOOP
# SOURCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
# REFERENCE: Cawley & Talbot (2010), JMLR 11, 2079-2107.
# MODIFICATION: Outer CV split is by climate zone (LOCO design) instead of
# standard KFold. Inner CV is RandomizedSearchCV on training zones only.
# Imputer is fit inside the loop on training data only per fold.
# XGBoost-specific: tree_method and device injected into best estimator
# after search to ensure GPU params are preserved in saved results.
# =============================================================================
print("\n" + "=" * 70)
print("NESTED LOCO CV — ONE TUNING RUN PER HELD-OUT ZONE")
print("=" * 70)
print("NOTE: The held-out zone is NEVER seen during hyperparameter search.")
print("      Resolves Reviewer #1 Comment #2 (data leakage in tuning).")
print("      Produces best_params_xgb_per_zone_revised.csv for Fix 4 & Fix 2.")
print("=" * 70)

all_zone_results = []
all_best_params  = {}
t_total_start    = time.time()

for held_out_zone in zones:
    print(f"\n{'─' * 60}")
    print(f"FOLD: Hold out [{held_out_zone}]")
    print(f"{'─' * 60}")

    train_df = df_full[df_full['koppen_zone'] != held_out_zone].copy()
    test_df  = df_full[df_full['koppen_zone'] == held_out_zone].copy()

    print(f"  Training zones : {[z for z in zones if z != held_out_zone]}")
    print(f"  Train samples  : {len(train_df):,}")
    print(f"  Test samples   : {len(test_df):,}")

    X_train_raw = train_df[FEATURE_COLS].values
    y_train     = train_df[TARGET_COL].values
    X_test_raw  = test_df[FEATURE_COLS].values
    y_test      = test_df[TARGET_COL].values

    # Imputation: fit ONLY on training data for this fold
    # SOURCE: https://scikit-learn.org/stable/modules/impute.html
    # MODIFICATION: Fresh imputer per fold prevents leakage of imputation
    # statistics from held-out zone. Identical pattern to Script 14.
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train_raw)
    X_test  = imputer.transform(X_test_raw)

    # Baseline (default params) evaluated on this fold for comparison
    # SOURCE: https://xgboost.readthedocs.io/en/stable/python/python_api.html
    # MODIFICATION: tree_method and device injected from runtime detection
    # above so baseline and tuned models use the same hardware path.
    baseline_model = xgb.XGBRegressor(
        **BASELINE_DEFAULTS,
        tree_method=TREE_METHOD,
        device=DEVICE,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    baseline_model.fit(X_train, y_train)
    y_pred_base   = baseline_model.predict(X_test)
    baseline_r2   = r2_score(y_test, y_pred_base)
    baseline_mae  = mean_absolute_error(y_test, y_pred_base)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
    print(f"  Baseline R²    : {baseline_r2:.4f}  MAE: {baseline_mae:.3f}  RMSE: {baseline_rmse:.3f}")

    # Inner RandomizedSearchCV — strictly on training data only
    # SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    # MODIFICATION: cv splits happen entirely within X_train (training zones).
    # X_test (held-out zone) only used AFTER search is complete.
    # n_jobs=1 when device='cuda': XGBoost GPU handles parallelism internally;
    # setting n_jobs=-1 with CUDA causes thread contention.
    xgb_base = xgb.XGBRegressor(
        tree_method=TREE_METHOD,
        device=DEVICE,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=(1 if DEVICE == 'cuda' else -1)
    )

    inner_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=N_INNER_CV,
        scoring='r2',
        n_jobs=(1 if DEVICE == 'cuda' else -1),
        verbose=0,
        random_state=RANDOM_STATE,
        return_train_score=True
    )

    print(f"  Running inner search ({N_ITER_SEARCH} iter x {N_INNER_CV}-fold)...", end=" ", flush=True)
    t_fold_start = time.time()
    inner_search.fit(X_train, y_train)
    t_fold = time.time() - t_fold_start
    print(f"done in {t_fold/60:.1f} min")

    best_params_fold = inner_search.best_params_
    best_cv_r2_fold  = inner_search.best_score_
    print(f"  Best inner CV R²: {best_cv_r2_fold:.4f}")

    # Evaluate best model on held-out zone (first and only time test is used)
    best_model   = inner_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    tuned_r2     = r2_score(y_test, y_pred_tuned)
    tuned_mae    = mean_absolute_error(y_test, y_pred_tuned)
    tuned_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    bias         = float(np.mean(y_pred_tuned - y_test))

    print(f"  Tuned   R²     : {tuned_r2:.4f}  MAE: {tuned_mae:.3f}  RMSE: {tuned_rmse:.3f}")
    print(f"  R² gain        : {tuned_r2 - baseline_r2:+.4f}")

    all_best_params[held_out_zone] = best_params_fold

    all_zone_results.append({
        'held_out_zone'   : held_out_zone,
        'train_n'         : len(train_df),
        'test_n'          : len(test_df),
        'baseline_r2'     : baseline_r2,
        'baseline_mae'    : baseline_mae,
        'baseline_rmse'   : baseline_rmse,
        'best_inner_cv_r2': best_cv_r2_fold,
        'tuned_r2'        : tuned_r2,
        'tuned_mae'       : tuned_mae,
        'tuned_rmse'      : tuned_rmse,
        'bias'            : bias,
        'r2_gain'         : tuned_r2 - baseline_r2,
        'mae_improvement' : baseline_mae - tuned_mae,
        'fold_time_min'   : t_fold / 60,
        **{f'param_{k}': v for k, v in best_params_fold.items()}
    })

t_total = time.time() - t_total_start

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("NESTED CV TUNING SUMMARY — XGBOOST")
print("=" * 70)
results_df = pd.DataFrame(all_zone_results)
print(f"{'Zone':<14} {'Baseline R²':>11} {'Tuned R²':>9} {'Gain':>7} {'MAE':>8} {'RMSE':>8}")
print("─" * 65)
for _, row in results_df.iterrows():
    print(f"{row['held_out_zone']:<14} {row['baseline_r2']:>11.4f} "
          f"{row['tuned_r2']:>9.4f} {row['r2_gain']:>+7.4f} "
          f"{row['tuned_mae']:>8.3f} {row['tuned_rmse']:>8.3f}")
print("─" * 65)
print(f"{'MEAN':<14} {results_df['baseline_r2'].mean():>11.4f} "
      f"{results_df['tuned_r2'].mean():>9.4f} "
      f"{results_df['r2_gain'].mean():>+7.4f} "
      f"{results_df['tuned_mae'].mean():>8.3f} "
      f"{results_df['tuned_rmse'].mean():>8.3f}")
print(f"\nTotal tuning time: {t_total/60:.1f} minutes")

# =============================================================================
# SAVE RESULTS
# _revised suffix preserves original files per project convention.
# best_params_xgb_per_zone_revised.csv is the primary output — consumed by:
#   Fix 4: 13_baseline_training_GPU.py (replaces hardcoded XGBoost dict)
#   Fix 2: 19_shap_importance.py (tuned XGBoost for SHAP consistency check)
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df.to_csv(
    RESULTS_DIR / "xgboost_tuning_results_revised.csv", index=False
)
print("  Saved: xgboost_tuning_results_revised.csv")

params_df = pd.DataFrame([{'zone': z, **p} for z, p in all_best_params.items()])
params_df.to_csv(
    RESULTS_DIR / "best_params_xgb_per_zone_revised.csv", index=False
)
print("  Saved: best_params_xgb_per_zone_revised.csv  ← consumed by Fix 4 & Fix 2")

pd.DataFrame({
    'Metric'  : ['R2', 'MAE', 'RMSE'],
    'Baseline': [results_df['baseline_r2'].mean(),
                 results_df['baseline_mae'].mean(),
                 results_df['baseline_rmse'].mean()],
    'Tuned'   : [results_df['tuned_r2'].mean(),
                 results_df['tuned_mae'].mean(),
                 results_df['tuned_rmse'].mean()],
}).to_csv(
    RESULTS_DIR / "xgboost_tuning_impact_revised.csv", index=False
)
print("  Saved: xgboost_tuning_impact_revised.csv")

# =============================================================================
# NEXT STEPS
# =============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("1. Verify best_params_xgb_per_zone_revised.csv — check R² per zone.")
print("2. Paste full terminal output to Perplexity for interpretation.")
print("3. Fix 4: Script 13 will load best_params_xgb_per_zone_revised.csv")
print("   so each LOCO fold uses zone-specific tuned XGBoost parameters.")
print("4. Fix 2: Script 19 will load Subtropical params from this CSV")
print("   to replace the untuned RandomForest in the SHAP consistency check.")
print("\nThis completes Fix 1 of 4 for PLOS ONE Revision PONE-D-26-16592.")
print("=" * 70)
print("TUNING COMPLETE")
print("=" * 70)