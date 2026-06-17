"""
Spatial Baseline Comparison: IDW and RBF vs LightGBM
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

PURPOSE (Reviewer Task C):
==========================
Reviewer requested comparison against purely spatial interpolation baselines
to establish whether the machine learning model provides skill beyond simple
geographic interpolation. Two classical spatial methods are compared against
the canonical LightGBM under the same LOCO-CV design:

  1. Inverse Distance Weighting (IDW) — spatial only, p=2
  2. Radial Basis Function (RBF, thin-plate spline) — spatial only

Both baselines use ONLY (lat, lon) per month-year to interpolate PM2.5
from training zone observations to test zone locations. No meteorological
or temporal features are used by the baselines — this is the most
conservative possible comparison.

IMPORTANT DESIGN NOTE — LOCO for spatial interpolation:
Spatial interpolation across a held-out climate zone is genuinely hard:
training points are geographically clustered in OTHER zones, with a spatial
gap where the test zone sits. This is the correct and honest evaluation.
It directly answers whether LightGBM's advantage comes from learning
physical relationships vs. just interpolating from nearby training points.

CODE SOURCES & ATTRIBUTION:
===========================
1. Inverse Distance Weighting (IDW):
   SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC4199009/
   REFERENCE: Lu & Wong (2008). An adaptive inverse-distance weighting
   spatial interpolation technique. Computers & Geosciences, 34(9),
   1044-1055. https://doi.org/10.1016/j.cageo.2007.07.010
   MODIFICATION: Implemented from scratch using numpy vectorised distance
   computation (no external library). p=2 (standard). Clipped to
   train set min/max to prevent extrapolation artefacts. Monthly subsets
   used to respect temporal structure (interpolate within same month-year).

2. RBF Interpolation (thin-plate spline):
   SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
   REFERENCE: Fasshauer, G.E. (2007). Meshfree Approximation Methods
   with MATLAB. World Scientific.
   MODIFICATION: neighbors=50 to limit memory use on large monthly subsets
   (full RBF matrix would be n² for n=40,000+ training points — infeasible).
   kernel='thin_plate_spline' (default, well-suited for smooth spatial fields).
   smoothing=0.1 added for numerical stability with scattered data.
   Monthly subsets used for temporal consistency with IDW.

3. LOCO cross-validation design:
   SOURCE: https://scikit-learn.org/stable/modules/cross_validation.html
   REFERENCE: Roberts et al. (2017). Cross-validation strategies for data
   with temporal, spatial, hierarchical, or phylogenetic structure.
   Ecography, 40(8), 913-929. https://doi.org/10.1111/ecog.02881
   MODIFICATION: Outer fold = climate zone holdout. Spatial baselines
   receive only lat/lon as input — no meteorological features.

4. LightGBM comparison baseline loaded dynamically:
   SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
   MODIFICATION: Reads loco_cv_results_revised.csv to avoid hardcoded values.

HARDWARE: Standard CPU (no GPU needed for spatial interpolation).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RBFInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
LOCO_RESULTS_FILE = RESULTS_DIR / "loco_cv_results_revised.csv"

TARGET_COL = 'pm25'

# IDW power parameter — p=2 is standard for spatial interpolation
# SOURCE: Lu & Wong (2008). Computers & Geosciences, 34(9), 1044-1055.
IDW_POWER = 2

# RBF neighbors — limits memory use (full n×n matrix infeasible for n>40k)
# SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
RBF_NEIGHBORS = 50
RBF_SMOOTHING = 0.1   # numerical stability for scattered data

print("=" * 70)
print("SPATIAL BASELINE COMPARISON: IDW + RBF vs LightGBM (Task C)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print("Baselines: IDW (p=2) and RBF (thin-plate spline)")
print("Design   : LOCO-CV — baselines interpolate from training zones only")
print("Input    : lat, lon only (no meteorology, no temporal features)")
print("=" * 70)

# =============================================================================
# IDW IMPLEMENTATION
# SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC4199009/
# REFERENCE: Lu & Wong (2008). Computers & Geosciences, 34(9), 1044-1055.
# MODIFICATION: Vectorised numpy. Clipped to train range. Monthly subsets.
# =============================================================================
def idw_predict(train_coords, train_values, test_coords, power=2):
    """
    Inverse Distance Weighting interpolation.
    train_coords : (n, 2) array of [lat, lon]
    train_values : (n,)   array of PM2.5 values
    test_coords  : (m, 2) array of [lat, lon]
    Returns      : (m,)   array of predicted PM2.5
    SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC4199009/
    MODIFICATION: Handles coincident points (distance=0) by returning
    exact training value. Clipped to [min, max] of training values
    to prevent extrapolation artefacts at zone boundaries.
    """
    # Euclidean distance in lat/lon degrees (consistent with training scale)
    diff    = test_coords[:, np.newaxis, :] - train_coords[np.newaxis, :, :]
    dists   = np.sqrt((diff ** 2).sum(axis=2))  # (m, n)

    # Handle exact coincident points — return exact value
    exact   = np.where(dists == 0)
    weights = 1.0 / np.where(dists == 0, 1e-10, dists) ** power
    w_sum   = weights.sum(axis=1, keepdims=True)
    preds   = (weights * train_values[np.newaxis, :]).sum(axis=1) / w_sum.squeeze()

    # Replace exact-match points with their training value
    for i, j in zip(exact[0], exact[1]):
        preds[i] = train_values[j]

    # Clip to training range — prevents wild extrapolation at zone edges
    preds = np.clip(preds, train_values.min(), train_values.max())
    return preds


# =============================================================================
# MONTHLY INTERPOLATION WRAPPER
# MODIFICATION: Spatial interpolation applied per month-year subset to respect
# temporal structure. Predicting 2020-Jan from 2020-Jan training points only.
# This is more honest than ignoring time (all years pooled) because PM2.5
# has strong seasonal and interannual variation.
# =============================================================================
def predict_by_month(train_df, test_df, method='idw'):
    """
    Apply spatial interpolation per month-year group.
    Interpolates from training zone lat/lon to test zone lat/lon
    within each (year, month) time step.
    """
    preds_all  = np.full(len(test_df), np.nan)
    test_df    = test_df.reset_index(drop=True)

    for (yr, mo), test_grp in test_df.groupby(['year', 'month']):
        # Training points for same month-year
        train_grp = train_df[
            (train_df['year'] == yr) & (train_df['month'] == mo)
        ]
        if len(train_grp) < 3:
            # Fallback: use all training months if too few for this month-year
            train_grp = train_df[train_df['month'] == mo]
        if len(train_grp) < 3:
            # Final fallback: global training mean for this time step
            preds_all[test_grp.index] = train_df[TARGET_COL].median()
            continue

        train_coords = train_grp[['lat', 'lon']].values
        train_vals   = train_grp[TARGET_COL].values
        test_coords  = test_grp[['lat', 'lon']].values

        if method == 'idw':
            # SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC4199009/
            preds = idw_predict(train_coords, train_vals,
                                test_coords, power=IDW_POWER)
        elif method == 'rbf':
            # SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/
            #         scipy.interpolate.RBFInterpolator.html
            # MODIFICATION: neighbors=50 for memory efficiency, smoothing=0.1
            # for numerical stability with clustered training points.
            try:
                rbf   = RBFInterpolator(
                    train_coords, train_vals,
                    neighbors=min(RBF_NEIGHBORS, len(train_coords) - 1),
                    kernel='thin_plate_spline',
                    smoothing=RBF_SMOOTHING
                )
                preds = rbf(test_coords)
                # Clip to training range — same as IDW
                preds = np.clip(preds, train_vals.min(), train_vals.max())
            except Exception:
                # Fallback to IDW if RBF fails (e.g. singular matrix)
                preds = idw_predict(train_coords, train_vals,
                                    test_coords, power=IDW_POWER)
        else:
            raise ValueError(f"Unknown method: {method}")

        preds_all[test_grp.index] = preds

    return preds_all


# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
zones   = sorted(df_full['koppen_zone'].unique())
print(f"  Total samples : {len(df_full):,}")
print(f"  Climate zones : {zones}")
print(f"  Year range    : {df_full['year'].min()}–{df_full['year'].max()}")

# =============================================================================
# LOCO-CV SPATIAL BASELINE LOOP
# SOURCE: Roberts et al. (2017). Ecography, 40(8), 913-929.
# MODIFICATION: IDW and RBF both run under identical LOCO folds for
# direct comparison with LightGBM results from script 13.
# =============================================================================
results = []

for test_zone in zones:
    print(f"\n{'─' * 60}")
    print(f"Held-out zone: {test_zone}")
    print(f"{'─' * 60}")

    train_data = df_full[df_full['koppen_zone'] != test_zone].copy()
    test_data  = df_full[df_full['koppen_zone'] == test_zone].copy()
    print(f"  Train: {len(train_data):,}  |  Test: {len(test_data):,}")

    y_test = test_data[TARGET_COL].values

    for method_name, method_key in [('IDW (p=2)', 'idw'),
                                     ('RBF (thin-plate)', 'rbf')]:
        print(f"  {method_name}...", end=" ", flush=True)
        try:
            y_pred = predict_by_month(train_data, test_data, method=method_key)

            # Replace any remaining NaN with training median
            nan_mask = np.isnan(y_pred)
            if nan_mask.any():
                y_pred[nan_mask] = train_data[TARGET_COL].median()

            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            bias = float(np.mean(y_pred - y_test))
            print(f"R²={r2:.4f}  MAE={mae:.3f}  RMSE={rmse:.3f}  Bias={bias:+.3f}")

        except Exception as e:
            print(f"FAILED: {e}")
            r2 = mae = rmse = bias = np.nan

        results.append({
            'zone'  : test_zone,
            'model' : method_name,
            'r2'    : r2,
            'mae'   : mae,
            'rmse'  : rmse,
            'bias'  : bias,
        })

# =============================================================================
# LOAD LIGHTGBM LOCO BASELINE FOR COMPARISON
# MODIFICATION: Dynamic load from script 13 output — no hardcoded values.
# =============================================================================
print(f"\n{'─' * 60}")
print("Loading LightGBM LOCO results for comparison...")
if not LOCO_RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"LOCO results not found: {LOCO_RESULTS_FILE}\n"
        "Run 13_baseline_training_GPU.py first."
    )

loco_df  = pd.read_csv(LOCO_RESULTS_FILE)
lgb_rows = loco_df[loco_df['model'].str.startswith('LightGBM')].copy()

for _, row in lgb_rows.iterrows():
    results.append({
        'zone'  : row['zone'],
        'model' : 'LightGBM (zone-tuned)',
        'r2'    : row['r2'],
        'mae'   : row['mae'],
        'rmse'  : row['rmse'],
        'bias'  : np.nan,
    })

# =============================================================================
# SUMMARY
# =============================================================================
df_results = pd.DataFrame(results)
summary    = df_results.groupby('model')[['r2', 'mae', 'rmse']].mean()

print("\n" + "=" * 70)
print("SPATIAL BASELINE vs LIGHTGBM — MEAN ACROSS ALL ZONES")
print("=" * 70)
print(f"\n  {'Model':<25} {'Mean R²':>9} {'Mean MAE':>10} {'Mean RMSE':>11}")
print(f"  {'─'*58}")
for model, row in summary.sort_values('r2', ascending=False).iterrows():
    print(f"  {model:<25} {row['r2']:>9.4f} {row['mae']:>10.3f} {row['rmse']:>11.3f}")

# LightGBM advantage over each baseline
lgb_mean_r2 = summary.loc['LightGBM (zone-tuned)', 'r2']
print(f"\n  {'─'*58}")
print(f"  LIGHTGBM ADVANTAGE OVER SPATIAL BASELINES")
print(f"  {'─'*58}")
for model in ['IDW (p=2)', 'RBF (thin-plate)']:
    if model in summary.index:
        base_r2  = summary.loc[model, 'r2']
        gain     = lgb_mean_r2 - base_r2
        gain_pct = gain / abs(base_r2) * 100 if base_r2 != 0 else np.nan
        print(f"  vs {model:<22}: ΔR² = {gain:+.4f} ({gain_pct:+.1f}%)")

# Per-zone table
print(f"\n  {'─'*58}")
print(f"  PER-ZONE R² BY METHOD")
print(f"  {'─'*58}")
pivot = df_results.pivot_table(
    index='zone', columns='model', values='r2'
).round(4)
print(pivot.to_string())

# =============================================================================
# SAVE
# =============================================================================
out_path = RESULTS_DIR / "spatial_baseline_results.csv"
df_results.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

summary_path = RESULTS_DIR / "spatial_baseline_summary.csv"
summary.reset_index().to_csv(summary_path, index=False)
print(f"  Saved: {summary_path}")

# =============================================================================
# INTERPRETATION
# =============================================================================
idw_r2 = summary.loc['IDW (p=2)', 'r2'] if 'IDW (p=2)' in summary.index else np.nan
rbf_r2 = summary.loc['RBF (thin-plate)', 'r2'] if 'RBF (thin-plate)' in summary.index else np.nan

print("\n" + "=" * 70)
print("INTERPRETATION FOR REVIEWER RESPONSE")
print("=" * 70)
print(f"  LightGBM R²        : {lgb_mean_r2:.4f}")
print(f"  IDW R²             : {idw_r2:.4f}")
print(f"  RBF R²             : {rbf_r2:.4f}")
print(f"  LightGBM vs IDW    : {lgb_mean_r2 - idw_r2:+.4f}")
print(f"  LightGBM vs RBF    : {lgb_mean_r2 - rbf_r2:+.4f}")

best_baseline = max(idw_r2, rbf_r2)
if lgb_mean_r2 > best_baseline + 0.05:
    print("\n  Verdict: LightGBM substantially outperforms spatial interpolation.")
    print("  The ML model learns physical PM2.5 relationships beyond pure")
    print("  geographic proximity. Geographic coordinates in LightGBM encode")
    print("  emission gradients, not just spatial autocorrelation.")
elif lgb_mean_r2 > best_baseline:
    print("\n  Verdict: LightGBM modestly outperforms spatial interpolation.")
    print("  Some advantage from meteorological and temporal features,")
    print("  but geographic structure is the dominant signal.")
else:
    print("\n  Verdict: Spatial interpolation matches or exceeds LightGBM.")
    print("  Review design — this would indicate spatial autocorrelation")
    print("  is the primary driver, not learned physical relationships.")

print("\n" + "=" * 70)
print("SPATIAL BASELINE COMPLETE")
print("=" * 70)
print(f"\nOutputs in: {RESULTS_DIR}")
print("  spatial_baseline_results.csv")
print("  spatial_baseline_summary.csv")
print("=" * 70)
