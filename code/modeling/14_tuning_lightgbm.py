"""
LightGBM Hyperparameter Tuning via Nested LOCO Cross-Validation
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (v3 — PLOS ONE Major Revision):
================================================
- CRITICAL FIX: Replaced single-zone tuning with nested CV per LOCO fold.
  Previous version tuned on Subtropical zone then used Subtropical as the
  held-out LOCO test fold — this constitutes data leakage that invalidates
  performance metrics. Each zone now has its own independently tuned params.
- Paths updated from F: drive to D: drive (active working copy).
- Output files use _revised suffix to preserve original results.

CODE SOURCES & ATTRIBUTION:
===========================
1. Nested Cross-Validation to prevent leakage:
   SOURCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
   REFERENCE: Cawley, G.C. & Talbot, N.L.C. (2010). On Over-fitting in Model
   Selection and Subsequent Selection Bias in Performance Evaluation.
   Journal of Machine Learning Research, 11, 2079-2107.
   MODIFICATION: Outer loop is LOCO zone split (leave-one-zone-out); inner
   loop is RandomizedSearchCV strictly on remaining training zones. The
   held-out zone is never seen during tuning.

2. RandomizedSearchCV:
   SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
   REFERENCE: Bergstra & Bengio (2012). Random Search for Hyper-Parameter
   Optimization. Journal of Machine Learning Research, 13, 281-305.
   MODIFICATION: n_iter reduced from 50 to 30 per fold (5 folds x 30 = 150
   total fits) to keep runtime manageable. cv=3 retained for inner splits.

3. LightGBM Parameters:
   SOURCE: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
   REFERENCE: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree. NeurIPS 2017.
   MODIFICATION: Search space unchanged from v2. colsample_bytree kept at
   0.8-1.0 given reduced feature count (8 features).

4. SimpleImputer fit-on-train-only pattern:
   SOURCE: https://scikit-learn.org/stable/modules/impute.html
   MODIFICATION: Imputer instantiated fresh per fold and fit only on training
   zones to prevent leakage of imputation statistics from held-out zone.

HARDWARE: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
ESTIMATED RUNTIME: ~45-60 min (5 folds x 30 iterations x 3-fold inner CV)
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
import lightgbm as lgb

# =============================================================================
# CONFIGURATION
# Paths verified against D:\PM25_Satellite_Research folder inventory
# =============================================================================
SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR    = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"

# blh removed: 100% missing -> constant after imputation, zero signal.
# wind_speed removed: collinear with u10/v10 (reviewer multicollinearity concern).
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'

# MODIFIED: n_iter reduced from 50 to 30 per fold for nested CV runtime.
# Bergstra & Bengio (2012) show 30-60 random trials capture most benefit
# of exhaustive search for moderate parameter spaces.
N_ITER_SEARCH = 30
N_INNER_CV    = 3
RANDOM_STATE  = 42

print("=" * 70)
print("LIGHTGBM HYPERPARAMETER TUNING — NESTED LOCO CV (v3)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print("Method  : Nested LOCO CV (outer=zone holdout, inner=RandomizedSearchCV)")
print("Fix     : Tuning now strictly within training folds — leakage removed")
print(f"Features: {FEATURE_COLS}")
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
# SOURCE: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# MODIFICATION: Same space as v2 — only the CV structure changed.
# =============================================================================
param_dist = {
    'n_estimators'     : [100, 200, 300, 400, 500],
    'max_depth'        : [4, 5, 6, 7, 8, 10],
    'learning_rate'    : [0.01, 0.02, 0.05, 0.1, 0.15],
    'subsample'        : [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
    'num_leaves'       : [20, 31, 50, 70],
    'min_child_samples': [10, 20, 30, 50],
}

BASELINE_DEFAULTS = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'num_leaves': 31, 'min_child_samples': 20
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
# =============================================================================
print("\n" + "=" * 70)
print("NESTED LOCO CV — ONE TUNING RUN PER HELD-OUT ZONE")
print("=" * 70)
print("NOTE: The held-out zone is NEVER seen during hyperparameter search.")
print("      This resolves Reviewer #1 Comment #2 (data leakage in tuning).")
print("=" * 70)

all_zone_results = []
all_best_params  = {}
t_total_start    = time.time()

for held_out_zone in zones:
    print(f"\n{'─' * 60}")
    print(f"FOLD: Hold out [{held_out_zone}]")
    print(f"{'─' * 60}")

    # Split: training zones vs held-out zone
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
    # statistics from held-out zone into training data.
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train_raw)
    X_test  = imputer.transform(X_test_raw)

    # Baseline (default params) evaluated on this fold for comparison
    baseline_model = lgb.LGBMRegressor(
        **BASELINE_DEFAULTS, random_state=RANDOM_STATE, verbose=-1
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
    # X_test (held-out zone) is only used AFTER search is complete.
    lgb_base = lgb.LGBMRegressor(
        verbose=-1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    inner_search = RandomizedSearchCV(
        estimator=lgb_base,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=N_INNER_CV,
        scoring='r2',
        n_jobs=-1,
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
print("NESTED CV TUNING SUMMARY")
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
# _revised suffix preserves original files per project convention
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df.to_csv(RESULTS_DIR / "lightgbm_tuning_results_revised.csv", index=False)
print("  Saved: lightgbm_tuning_results_revised.csv")

params_df = pd.DataFrame([{'zone': z, **p} for z, p in all_best_params.items()])
params_df.to_csv(RESULTS_DIR / "best_params_per_zone_revised.csv", index=False)
print("  Saved: best_params_per_zone_revised.csv")

pd.DataFrame({
    'Metric'  : ['R2', 'MAE', 'RMSE'],
    'Baseline': [results_df['baseline_r2'].mean(),
                 results_df['baseline_mae'].mean(),
                 results_df['baseline_rmse'].mean()],
    'Tuned'   : [results_df['tuned_r2'].mean(),
                 results_df['tuned_mae'].mean(),
                 results_df['tuned_rmse'].mean()],
}).to_csv(RESULTS_DIR / "tuning_impact_visualization_revised.csv", index=False)
print("  Saved: tuning_impact_visualization_revised.csv")

# =============================================================================
# NEXT STEPS
# =============================================================================
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("1. Verify lightgbm_tuning_results_revised.csv — check R² per zone.")
print("2. Paste full output to Perplexity for interpretation.")
print("3. Script 13 will be updated to load best_params_per_zone_revised.csv")
print("   so each LOCO fold uses its own independently tuned parameters.")
print("\nThis fully addresses Reviewer #1 Comment #2 (data leakage fix).")
print("=" * 70)
print("TUNING COMPLETE")
print("=" * 70)