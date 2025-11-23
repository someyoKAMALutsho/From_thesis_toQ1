"""
Temporal Validation: Train-Test Time Split
===========================================
Alternative to LOCO: Train on earlier years (2019-2021), test on later (2022-2023)


Why: Detects temporal drift, data quality changes, or aerosol regime shifts
Source: Standard time-series ML practice (no specific paper needed)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\modeling")


INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"


FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10', 'blh']
TARGET_COL = 'pm25'


print("=" * 70)
print("TEMPORAL VALIDATION: CHRONOLOGICAL SPLIT")
print("=" * 70)


df_full = pd.read_parquet(INPUT_FILE)


# Extract year from month if needed (or assume year column exists)
# Assuming year column is pre-computed
train_data = df_full[df_full['year'].isin([2019, 2020, 2021])].copy()
test_data = df_full[df_full['year'].isin([2022, 2023])].copy()


print(f"Train period (2019-2021): {len(train_data):,} samples")
print(f"Test period (2022-2023): {len(test_data):,} samples")


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
print("\nTraining LightGBM (2019-2021 data)...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=10, learning_rate=0.15,
    num_leaves=70, subsample=0.6, colsample_bytree=1.0,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train[ALL_FEATURES], y_train)


# Evaluate on future years
y_pred = lgb_model.predict(X_test[ALL_FEATURES])


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print(f"\nTest on 2022-2023 data:")
print(f"  MAE: {mae:.3f} µg/m³")
print(f"  RMSE: {rmse:.3f}")
print(f"  R²: {r2:.4f}")


# Compare to LOCO R² on same zones
loco_r2_avg = 0.341  # Tuned LightGBM LOCO average
temporal_drop = (loco_r2_avg - r2) / loco_r2_avg * 100


print(f"\nComparison:")
print(f"  LOCO R² (across all zones): {loco_r2_avg:.4f}")
print(f"  Temporal R² (2022-2023): {r2:.4f}")
print(f"  Degradation: {temporal_drop:+.1f}%")


if abs(temporal_drop) > 20:
    print(f"\n⚠️  Significant temporal drift detected!")
    print(f"   Possible causes:")
    print(f"   - Satellite retrieval algorithm changes")
    print(f"   - Aerosol regime shifts (climate/emissions)")
    print(f"   - MERRA-2 meteorology recalibration")
else:
    print(f"\n✓ Model stable over time")


print(f"\n✓ Temporal validation complete")


# ====================== NEW PART: Extract LOCO metrics from CSV and Save Validation Metrics to CSV =========================

# Load LOCO CV results, extract global average R2, MAE, RMSE for LightGBM
try:
    loco_csv = pd.read_csv(RESULTS_DIR / "loco_cv_results_gpu.csv")
    lgb_loco = loco_csv[loco_csv['model'].str.lower().str.contains('lightgbm')]
    loco_r2_avg = lgb_loco['r2'].mean()
    loco_mae_avg = lgb_loco['mae'].mean()
    loco_rmse_avg = lgb_loco['rmse'].mean()
except Exception as e:
    print(f"Could not auto-extract LOCO metrics: {e}")
    print("Using manual LOCO average values in results below.")
    loco_r2_avg = 0.341
    loco_mae_avg = None
    loco_rmse_avg = None

metrics = {
    'Metric': ['R2', 'MAE', 'RMSE'],
    'LOCO': [loco_r2_avg, loco_mae_avg, loco_rmse_avg],
    'Temporal': [r2, mae, rmse]
}
df = pd.DataFrame(metrics)
df.to_csv(RESULTS_DIR / "temporal_vs_loco_validation.csv", index=False)
print("\n✓ Saved: temporal_vs_loco_validation.csv")
