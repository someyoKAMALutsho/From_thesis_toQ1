"""
SHAP Feature Importance + Cross-Model Consistency + Partial Dependence
For PM2.5 Satellite Prediction Paper — PLOS ONE Revision (PONE-D-26-16592)

REVISION NOTES (v3 — PLOS ONE Major Revision):
===============================================
- CRITICAL FIX: LightGBM now uses zone-specific hyperparameters loaded from
  best_params_per_zone_revised.csv for all SHAP analyses. Previous version
  used a single hardcoded parameter set — inconsistent with nested CV tuning
  and Scripts 13/17. SHAP values from incorrectly-tuned models are not
  representative of the model actually used for prediction.
- CRITICAL FIX: Parts 1-3 (SHAP importance, cross-model consistency, PDP)
  now run across ALL zones, not only Subtropical. Per-zone SHAP figures
  are saved individually. Subtropical is retained as the primary example
  in manuscript figures but all zones are reported in supplementary.
- Output files use _revised suffix to preserve originals.
- Paths already dynamic from path-fix pass.

This script produces:
  shap_importance_{zone}_revised.csv / .png  (one per zone)
  shap_cross_model_consistency_revised.csv / .png
  shap_category_consistency_revised.csv
  pdp_dose_response_{zone}_revised.png  (one per zone)
  shap_by_zone_revised.csv
  geographic_dominance_heatmap_revised.csv

CODE SOURCES & ATTRIBUTION:
===========================
1. SHAP TreeExplainer:
   SOURCE: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
   REFERENCE: Lundberg & Lee (2017). A Unified Approach to Interpreting Model
   Predictions. NeurIPS 2017.
   MODIFICATION: Applied per zone using zone-specific tuned model.
   Background data = held-out test set (standard for SHAP global explanation).

2. Permutation Importance (cross-model consistency):
   SOURCE: https://scikit-learn.org/stable/modules/permutation_importance.html
   REFERENCE: Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
   MODIFICATION: n_repeats=10 for stability. Applied to test set only
   (fit strictly on training zones — no leakage).

3. Partial Dependence:
   SOURCE: https://scikit-learn.org/stable/modules/partial_dependence.html
   REFERENCE: Friedman (2001). Greedy Function Approximation: A Gradient
   Boosting Machine. Annals of Statistics, 29(5), 1189-1232.
   MODIFICATION: Applied to top-3 SHAP features per zone. grid_resolution=50
   for smooth curves. Uses held-out test set as background.

4. Zone-specific parameter loading:
   SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
   MODIFICATION: Same pattern as scripts 13 and 17 for consistency.

5. Köppen LOCO design:
   SOURCE: Roberts et al. (2017). Ecography, 40(8), 913-929.
   MODIFICATION: Imputer fit on training zones only per fold.

HARDWARE: NVIDIA GeForce RTX 3060 (Compute Capability 8.6)
NOTE: LightGBM runs on CPU (pip build without CUDA).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap'])
    import shap

from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
import lightgbm as lgb
import xgboost as xgb

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

# FEATURE SET LOCKED — 8 features empirically optimal.
FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10']
TARGET_COL   = 'pm25'
RANDOM_STATE = 42

CATEGORY_MAP = {
    'lat': 'Geographic', 'lon': 'Geographic',
    'year': 'Temporal',  'month': 'Temporal',
    't2m': 'Meteorology', 'ps': 'Meteorology',
    'u10': 'Meteorology', 'v10': 'Meteorology',
}

CATEGORY_COLORS = {
    'Geographic' : '#2166ac',
    'Meteorology': '#d6604d',
    'Temporal'   : '#4dac26',
}

print("=" * 70)
print("SHAP FEATURE IMPORTANCE ANALYSIS (v3)")
print("PLOS ONE Revision: PONE-D-26-16592")
print("=" * 70)
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print("Fix: Zone-specific LightGBM params used for all SHAP analyses")
print("Fix: All zones analysed (not only Subtropical)")
print("=" * 70)

# =============================================================================
# LOAD ZONE-SPECIFIC PARAMETERS
# SOURCE: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# MODIFICATION: Same loading pattern as scripts 13 and 17.
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
# HELPERS
# SOURCE: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
# MODIFICATION: Returns mean absolute SHAP values across background dataset.
# =============================================================================
def prepare_zone_data(df, test_zone):
    train = df[df['koppen_zone'] != test_zone].copy()
    test  = df[df['koppen_zone'] == test_zone].copy()
    X_tr  = train[FEATURE_COLS].copy()
    y_tr  = train[TARGET_COL].copy()
    X_te  = test[FEATURE_COLS].copy()
    y_te  = test[TARGET_COL].copy()
    imputer = SimpleImputer(strategy='median')
    X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=FEATURE_COLS)
    X_te = pd.DataFrame(imputer.transform(X_te),     columns=FEATURE_COLS)
    return X_tr, y_tr, X_te, y_te

def compute_shap_mean_abs(model, X):
    # Monkey-patch SHAP's UBJSON decoder to fix XGBoost base_score string/list parsing issues
    try:
        import shap.explainers._tree as shap_tree
        if not hasattr(shap_tree, '_patched_decode_ubjson_buffer'):
            original_decode = shap_tree.decode_ubjson_buffer
            def patched_decode(*args, **kwargs):
                result = original_decode(*args, **kwargs)
                try:
                    if "learner" in result:
                        learner = result["learner"]
                        if "learner_model_param" in learner:
                            param = learner["learner_model_param"]
                            if "base_score" in param:
                                bs = param["base_score"]
                                if isinstance(bs, str):
                                    param["base_score"] = bs.strip('[]')
                                elif isinstance(bs, list) and len(bs) > 0:
                                    param["base_score"] = str(bs[0])
                except Exception:
                    pass
                return result
            shap_tree.decode_ubjson_buffer = patched_decode
            shap_tree._patched_decode_ubjson_buffer = True
    except Exception:
        pass

    if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model.get_booster())
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    return np.abs(sv).mean(axis=0)

def plot_shap_bar(shap_df, zone, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [CATEGORY_COLORS[CATEGORY_MAP[f]] for f in shap_df['Feature']]
    ax.barh(shap_df['Feature'], shap_df['Relative Importance'], color=colors)
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title(f'SHAP Feature Importance — {zone} Zone (LightGBM)',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(handles=[
        Patch(color=v, label=k) for k, v in CATEGORY_COLORS.items()
    ], fontsize=9, title='Category')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN LOOP — All zones
# MODIFICATION: Runs Parts 1 (SHAP importance), 3 (PDP), and 4 (aggregation)
# for every zone. Part 2 (cross-model consistency) runs on Subtropical only
# as primary example — computationally expensive (permutation importance).
# All zones are covered in Part 4 (per-zone SHAP aggregation).
# =============================================================================
per_zone_feat = []
per_zone_cat  = []
consistency_zone = 'Subtropical'  # Primary example for cross-model figure

print("\n" + "=" * 70)
print("PARTS 1 + 3: SHAP IMPORTANCE AND PDP — ALL ZONES")
print("=" * 70)

for zone in zones:
    print(f"\n{'-' * 60}")
    print(f"Zone: {zone}")
    print(f"{'-' * 60}")

    X_tr, y_tr, X_te, y_te = prepare_zone_data(df_full, zone)

    # Train zone-specific LightGBM
    lgb_model = lgb.LGBMRegressor(**ZONE_PARAMS[zone])
    lgb_model.fit(X_tr, y_tr)
    r2_z = r2_score(y_te, lgb_model.predict(X_te))
    print(f"  LightGBM R²: {r2_z:.4f}")

    # --- PART 1: SHAP importance ---
    # SOURCE: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    print(f"  Computing SHAP values...", end=" ", flush=True)
    mean_shap = compute_shap_mean_abs(lgb_model, X_te)
    shap_df = pd.DataFrame({
        'Feature'            : FEATURE_COLS,
        'Mean |SHAP|'        : mean_shap,
        'Relative Importance': mean_shap / mean_shap.sum() * 100,
        'Category'           : [CATEGORY_MAP[f] for f in FEATURE_COLS],
        'Zone'               : zone,
    }).sort_values('Mean |SHAP|', ascending=False)
    print("done")

    shap_df.to_csv(
        RESULTS_DIR / f"shap_importance_{zone.lower()}_revised.csv", index=False
    )
    plot_shap_bar(
        shap_df, zone,
        RESULTS_DIR / f"shap_importance_{zone.lower()}_revised.png"
    )
    print(f"  Saved: shap_importance_{zone.lower()}_revised.png")

    # Top features for this zone
    top3 = shap_df.head(3)['Feature'].tolist()
    geo_pct = shap_df[shap_df['Category'] == 'Geographic']['Relative Importance'].sum()
    print(f"  Top 3: {top3}  |  Geographic %: {geo_pct:.1f}%")

    # --- PART 3: PDP for top 3 features ---
    # SOURCE: https://scikit-learn.org/stable/modules/partial_dependence.html
    # REFERENCE: Friedman (2001). Annals of Statistics, 29(5), 1189-1232.
    print(f"  Computing PDPs...", end=" ", flush=True)
    top3_idx = [FEATURE_COLS.index(f) for f in top3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat, feat_idx in zip(axes, top3, top3_idx):
        pd_result = partial_dependence(
            estimator=lgb_model,
            X=X_te,
            features=[feat_idx],
            grid_resolution=50
        )
        xs = pd_result['grid_values'][0]
        ys = pd_result['average'][0]
        ax.plot(xs, ys, color=CATEGORY_COLORS[CATEGORY_MAP[feat]], linewidth=2)
        ax.set_title(f'{feat} ({CATEGORY_MAP[feat]})',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel(feat)
        ax.set_ylabel('Predicted PM₂.₅ (µg m⁻³)')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Partial Dependence — Top Features ({zone} Zone)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"pdp_dose_response_{zone.lower()}_revised.png",
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print("done")

    # --- PART 4 data collection ---
    per_zone_feat.append(shap_df)
    cat_df = shap_df.groupby('Category')['Mean |SHAP|'].sum().reset_index()
    cat_df['Zone'] = zone
    cat_df['R2']   = r2_z
    per_zone_cat.append(cat_df)

    # Store lgb_model and data for cross-model consistency zone
    if zone == consistency_zone:
        lgb_consistency  = lgb_model
        X_tr_consistency = X_tr
        y_tr_consistency = y_tr
        X_te_consistency = X_te
        y_te_consistency = y_te
        shap_lgb_pct     = shap_df.set_index('Feature')['Relative Importance']

# =============================================================================
# PART 2 — Cross-model consistency (Subtropical as primary example)
# SOURCE: https://scikit-learn.org/stable/modules/permutation_importance.html
# REFERENCE: Breiman (2001). Machine Learning, 45(1), 5-32.
# MODIFICATION: RF fit strictly on training zones (no leakage).
#   Permutation importance evaluated on held-out test zone only.
# =============================================================================
print("\n" + "=" * 70)
print(f"PART 2: CROSS-MODEL CONSISTENCY — {consistency_zone} zone")
print("=" * 70)

# CRITICAL FIX (Fix 2): Replace untuned RandomForest + permutation_importance
# with tuned XGBoost + shap.TreeExplainer for cross-model consistency check.
# Using the same method (SHAP TreeExplainer) on both models eliminates the
# apples-to-oranges comparison between SHAP values and permutation importance.
# SOURCE: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
# REFERENCE: Lundberg & Lee (2017). NeurIPS 2017.
# REFERENCE: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
# MODIFICATION: XGBoost fitted on same training zones as LightGBM (no leakage).
#   Zone-specific tuned params loaded from best_params_xgb_per_zone_revised.csv.
#   SHAP TreeExplainer used for both models — methodologically consistent.

XGB_TUNED_PARAMS_FILE = RESULTS_DIR / "best_params_xgb_per_zone_revised.csv"
if not XGB_TUNED_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"XGBoost tuned params not found: {XGB_TUNED_PARAMS_FILE}\n"
        "Run 15_tuning_xgboost.py first."
    )
xgb_params_raw = pd.read_csv(XGB_TUNED_PARAMS_FILE)
xgb_consistency_row = xgb_params_raw[
    xgb_params_raw['zone'] == consistency_zone
].iloc[0]
xgb_consistency_params = {
    'n_estimators'    : int(xgb_consistency_row['n_estimators']),
    'max_depth'       : int(xgb_consistency_row['max_depth']),
    'learning_rate'   : float(xgb_consistency_row['learning_rate']),
    'subsample'       : float(xgb_consistency_row['subsample']),
    'colsample_bytree': float(xgb_consistency_row['colsample_bytree']),
    'min_child_weight': int(xgb_consistency_row['min_child_weight']),
    'reg_alpha'       : float(xgb_consistency_row['reg_alpha']),
    'reg_lambda'      : float(xgb_consistency_row['reg_lambda']),
    'random_state'    : RANDOM_STATE,
    'verbosity'       : 0,
    'tree_method'     : 'hist',
}

# GPU fallback for XGBoost consistency model
# SOURCE: https://xgboost.readthedocs.io/en/stable/gpu_support.html
# MODIFICATION: Same runtime detection as 15_tuning_xgboost.py.
try:
    _probe = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=1)
    _probe.fit(np.zeros((10, 2)), np.zeros(10))
    xgb_consistency_params['device'] = 'cuda'
    print("  XGBoost consistency model: GPU (cuda)")
except Exception:
    xgb_consistency_params['device'] = 'cpu'
    print("  XGBoost consistency model: CPU fallback")

print("  Training tuned XGBoost for consistency check...", end=" ", flush=True)
xgb_model = xgb.XGBRegressor(**xgb_consistency_params)
xgb_model.fit(X_tr_consistency, y_tr_consistency)
xgb_r2 = r2_score(y_te_consistency, xgb_model.predict(X_te_consistency))
print(f"done  (R²={xgb_r2:.4f})")

# SHAP TreeExplainer on tuned XGBoost — same method as LightGBM above
# SOURCE: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
# MODIFICATION: Both models now use identical SHAP-based importance method.
#   This directly addresses Reviewer #2 concern about consistency of
#   interpretation across models.
print("  Computing SHAP values (XGBoost)...", end=" ", flush=True)
xgb_mean_shap = compute_shap_mean_abs(xgb_model, X_te_consistency)
xgb_pct = pd.Series(
    xgb_mean_shap / xgb_mean_shap.sum() * 100,
    index=FEATURE_COLS
)
print("done")

consistency_df = pd.DataFrame({
    'Feature'            : FEATURE_COLS,
    'Category'           : [CATEGORY_MAP[f] for f in FEATURE_COLS],
    'LightGBM_SHAP_pct'  : shap_lgb_pct[FEATURE_COLS].values,
    'XGBoost_SHAP_pct'   : xgb_pct.values,
})
consistency_df['Mean_pct'] = consistency_df[
    ['LightGBM_SHAP_pct', 'XGBoost_SHAP_pct']
].mean(axis=1)
consistency_df['Std_pct']  = consistency_df[
    ['LightGBM_SHAP_pct', 'XGBoost_SHAP_pct']
].std(axis=1)
consistency_df['Rank_LGB'] = consistency_df['LightGBM_SHAP_pct'].rank(
    ascending=False).astype(int)
consistency_df['Rank_XGB'] = consistency_df['XGBoost_SHAP_pct'].rank(
    ascending=False).astype(int)
consistency_df = consistency_df.sort_values('Mean_pct', ascending=False)

consistency_df.to_csv(
    RESULTS_DIR / "shap_cross_model_consistency_revised.csv", index=False
)

print("\nCross-model consistency:")
print(consistency_df[[
    'Feature', 'Category', 'LightGBM_SHAP_pct',
    'XGBoost_SHAP_pct', 'Rank_LGB', 'Rank_XGB'
]].round(1).to_string(index=False))

cat_totals = consistency_df.groupby('Category')[
    ['LightGBM_SHAP_pct', 'XGBoost_SHAP_pct']
].sum()
cat_totals.to_csv(
    RESULTS_DIR / "shap_category_consistency_revised.csv"
)
print("\nCategory totals (%):")
print(cat_totals.round(1))

fig, ax = plt.subplots(figsize=(8, 5))
cat_totals.rename(columns={
    'LightGBM_SHAP_pct': 'LightGBM (SHAP)',
    'XGBoost_SHAP_pct' : 'XGBoost (SHAP)'
}).T.plot(kind='bar', ax=ax, width=0.7, colormap='Set2')
ax.set_title(
    f'Feature Category Importance — {consistency_zone} Zone\n'
    f'LightGBM SHAP vs XGBoost SHAP (tuned, nested LOCO-CV)',
    fontsize=11, fontweight='bold'
)
ax.set_ylabel('Total Relative Importance (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Model', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "shap_cross_model_consistency_revised.png",
    dpi=300, bbox_inches='tight'
)
plt.close()
print(f"  Saved: shap_cross_model_consistency_revised.png")

# =============================================================================
# PART 4 — Per-zone SHAP aggregation summary
# =============================================================================
print("\n" + "=" * 70)
print("PART 4: PER-ZONE SHAP AGGREGATION SUMMARY")
print("=" * 70)

feature_matrix = pd.concat(per_zone_feat, ignore_index=True)
heatmap_pivot  = feature_matrix.pivot_table(
    index='Feature', columns='Zone', values='Mean |SHAP|'
)
heatmap_pivot.to_csv(
    RESULTS_DIR / "geographic_dominance_heatmap_revised.csv"
)
print("  Saved: geographic_dominance_heatmap_revised.csv")

category_matrix = pd.concat(per_zone_cat, ignore_index=True)
shap_by_zone = category_matrix.pivot_table(
    index='Zone', columns='Category', values='Mean |SHAP|'
).reset_index()

for col in ['Geographic', 'Meteorology', 'Temporal']:
    if col not in shap_by_zone.columns:
        shap_by_zone[col] = 0.0

total = shap_by_zone[['Geographic', 'Meteorology', 'Temporal']].sum(axis=1)
for col in ['Geographic', 'Meteorology', 'Temporal']:
    shap_by_zone[f'{col}_pct'] = np.where(
        total > 0,
        (shap_by_zone[col] / total * 100).round(1), 0.0
    )

shap_by_zone.to_csv(RESULTS_DIR / "shap_by_zone_revised.csv", index=False)
print("  Saved: shap_by_zone_revised.csv")

print("\nSHAP category percentages by zone:")
print(shap_by_zone[[
    'Zone', 'Geographic_pct', 'Meteorology_pct', 'Temporal_pct'
]].to_string(index=False))

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE (v3)")
print("=" * 70)
print(f"\nOutputs in: {RESULTS_DIR}")
print("  shap_importance_{{zone}}_revised.csv / .png  (5 zones)")
print("  shap_cross_model_consistency_revised.csv / .png")
print("  shap_category_consistency_revised.csv")
print("  pdp_dose_response_{{zone}}_revised.png  (5 zones)")
print("  geographic_dominance_heatmap_revised.csv")
print("  shap_by_zone_revised.csv")
print("\nNext: Run 20_temporal_validation.py")
print("=" * 70)