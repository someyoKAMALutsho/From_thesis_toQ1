"""
SHAP (SHapley Additive exPlanations) Feature Importance
======================================================
Why SHAP > standard feature_importances:
- Provides global + local explanations
- Accounts for feature interactions
- Theoretically sound (game theory)


Source: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
Reference: Lundberg & Lee (2017). NIPS
"""


import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("Installing SHAP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'shap'])
    import shap
    HAS_SHAP = True


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb


DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\modeling")


INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"


FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10', 'blh']
TARGET_COL = 'pm25'


print("=" * 70)
print("SHAP FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)


df_full = pd.read_parquet(INPUT_FILE)


# Use Subtropical zone for demonstration (best model)
test_zone = 'Subtropical'
train_data = df_full[df_full['koppen_zone'] != test_zone].copy()
test_data = df_full[df_full['koppen_zone'] == test_zone].copy()


X_train = train_data[FEATURE_COLS].copy()
y_train = train_data[TARGET_COL].copy()
X_test = test_data[FEATURE_COLS].copy()
y_test = test_data[TARGET_COL].copy()


# Impute
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLS)
X_test = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLS)


X_train['wind_speed'] = np.sqrt(X_train['u10']**2 + X_train['v10']**2)
X_test['wind_speed'] = np.sqrt(X_test['u10']**2 + X_test['v10']**2)


ALL_FEATURES = FEATURE_COLS + ['wind_speed']


# Train tuned LightGBM
print(f"Training LightGBM on {test_zone}...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=10, learning_rate=0.15,
    num_leaves=70, subsample=0.6, colsample_bytree=1.0,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train[ALL_FEATURES], y_train)


# SHAP explainer
print("Computing SHAP values...")
# Source: [https://shap.readthedocs.io/en/latest/example_notebooks/explainers/tree_explainer/](https://shap.readthedocs.io/en/latest/example_notebooks/explainers/tree_explainer/)
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test[ALL_FEATURES])


# Mean absolute SHAP value = average impact per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': ALL_FEATURES,
    'Mean |SHAP|': mean_abs_shap,
    'Relative Importance': mean_abs_shap / mean_abs_shap.sum() * 100
}).sort_values('Mean |SHAP|', ascending=False)


print(f"\nFeature Importance (Subtropical zone):")
print(feature_importance_df.to_string(index=False))


# Save
feature_importance_df.to_csv(RESULTS_DIR / "shap_importance_subtropical.csv", index=False)


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_df.plot(x='Feature', y='Relative Importance', kind='barh', ax=ax, legend=False)
ax.set_title(f'SHAP Feature Importance - {test_zone} Zone', fontsize=12, fontweight='bold')
ax.set_xlabel('Relative Importance (%)')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_importance_subtropical.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: shap_importance_subtropical.png")
plt.close()


print(f"✓ SHAP analysis complete")


# ============================ NEW PART: SHAP BY ZONE AGGREGATION FOR ALL ZONES =============================
# This additional code generates per-zone summary CSVs and a heatmap matrix for figure-generation.

zones = ['Subtropical', 'Temperate', 'Boreal', 'Polar', 'Tropical']
category_map = {
    'lat': 'Geographic', 'lon': 'Geographic',
    'year': 'Temporal', 'month': 'Temporal',
    't2m': 'Meteorology', 'ps': 'Meteorology',
    'u10': 'Meteorology', 'v10': 'Meteorology',
    'blh': 'Meteorology', 'wind_speed': 'Meteorology'
}

per_zone_cat = []
per_zone_feat = []

for zone in zones:
    train_z = df_full[df_full['koppen_zone'] != zone].copy()
    test_z = df_full[df_full['koppen_zone'] == zone].copy()
    X_train_z = train_z[FEATURE_COLS].copy()
    y_train_z = train_z[TARGET_COL].copy()
    X_test_z = test_z[FEATURE_COLS].copy()
    y_test_z = test_z[TARGET_COL].copy()
    # Impute
    X_train_z = pd.DataFrame(imputer.fit_transform(X_train_z), columns=FEATURE_COLS)
    X_test_z = pd.DataFrame(imputer.transform(X_test_z), columns=FEATURE_COLS)
    X_train_z['wind_speed'] = np.sqrt(X_train_z['u10']**2 + X_train_z['v10']**2)
    X_test_z['wind_speed'] = np.sqrt(X_test_z['u10']**2 + X_test_z['v10']**2)
    # Train
    lgb_m = lgb.LGBMRegressor(
        n_estimators=200, max_depth=10, learning_rate=0.15,
        num_leaves=70, subsample=0.6, colsample_bytree=1.0,
        random_state=42, verbose=-1
    )
    lgb_m.fit(X_train_z[ALL_FEATURES], y_train_z)
    explainer_z = shap.TreeExplainer(lgb_m)
    shap_vals = explainer_z.shap_values(X_test_z[ALL_FEATURES])
    mean_abs_shap_z = np.abs(shap_vals).mean(axis=0)
    feat_df = pd.DataFrame({
        'Feature': ALL_FEATURES,
        'Mean|SHAP|': mean_abs_shap_z,
        'Zone': zone
    })
    per_zone_feat.append(feat_df)
    # Calculate per-category sums
    feat_df['Category'] = feat_df['Feature'].map(category_map)
    cat_df = feat_df.groupby('Category')['Mean|SHAP|'].sum().reset_index()
    cat_df['Zone'] = zone
    per_zone_cat.append(cat_df)

# Aggregate per-zone feature means as matrix for heatmap
feature_matrix = pd.concat(per_zone_feat)
feature_matrix_out = feature_matrix.pivot(index='Feature', columns='Zone', values='Mean|SHAP|')
feature_matrix_out.to_csv(RESULTS_DIR / "geographic_dominance_heatmap.csv")

# Aggregate per-zone category means as summary table
category_matrix = pd.concat(per_zone_cat)
shap_by_zone_out = category_matrix.pivot(index='Zone', columns='Category', values='Mean|SHAP|').reset_index()
shap_by_zone_out.to_csv(RESULTS_DIR / "shap_by_zone.csv", index=False)

print("✓ Saved: shap_by_zone.csv (category SHAP by zone table)")
print("✓ Saved: geographic_dominance_heatmap.csv (feature SHAP by zone matrix)")
