"""
Ensemble Stacking: Meta-Learner Approach
=========================================
Level 0 (Base Models): LightGBM, XGBoost, Random Forest (trained separately)
Level 1 (Meta-Model): Linear Regression learns optimal weights

Expected Gain: +5–10% R² improvement
Source: Wolpert (1992). Stacking; Scikit-learn API

Reference: Zhou (2012). Ensemble Methods: Foundations and Algorithms. CRC Press.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\modeling")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"

FEATURE_COLS = ['lat', 'lon', 'year', 'month', 't2m', 'ps', 'u10', 'v10', 'blh']
TARGET_COL = 'pm25'

print("=" * 70)
print("ENSEMBLE STACKING: META-LEARNER APPROACH")
print("=" * 70)

df_full = pd.read_parquet(INPUT_FILE)
zones = sorted(df_full['koppen_zone'].unique())

results = {
    'zone': [], 'model': [], 'mae': [], 'rmse': [], 'r2': [],
    'train_time_sec': []
}

print("\nLOCO CROSS-VALIDATION WITH STACKING\n")

for test_zone in zones:
    print(f"Testing on {test_zone}...")
    
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
    
    # ===== LEVEL 0: BASE MODELS =====
    # Source: https://scikit-learn.org/stable/modules/ensemble.html#stacking
    
    base_models = [
        ('lgb', lgb.LGBMRegressor(
            n_estimators=200, max_depth=10, learning_rate=0.15,
            num_leaves=70, subsample=0.6, colsample_bytree=1.0,
            random_state=42, verbose=-1
        )),
        ('xgb', xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )),
        ('rf', RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )),
    ]
    
    # ===== LEVEL 1: META-MODEL =====
    # Source: https://scikit-learn.org/stable/modules/ensemble.html#stacking
    meta_model = LinearRegression()
    
    # ===== STACKING REGRESSOR =====
    print(f"  Building stacking ensemble...", end=" ", flush=True)
    t0 = time.time()
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5  # 5-fold CV for base model meta-features
    )
    
    stacking_model.fit(X_train[ALL_FEATURES], y_train)
    y_pred_stack = stacking_model.predict(X_test[ALL_FEATURES])
    t_stack = time.time() - t0
    
    mae = mean_absolute_error(y_test, y_pred_stack)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    r2 = r2_score(y_test, y_pred_stack)
    
    print(f"✓ MAE={mae:.3f}, R²={r2:.4f} ({t_stack:.1f}s)")
    
    results['zone'].append(test_zone)
    results['model'].append('Stacking (LGB+XGB+RF)')
    results['mae'].append(mae)
    results['rmse'].append(rmse)
    results['r2'].append(r2)
    results['train_time_sec'].append(t_stack)

df_stacking = pd.DataFrame(results)
df_stacking.to_csv(RESULTS_DIR / "stacking_results.csv", index=False)

print(f"\n" + "=" * 70)
print("STACKING RESULTS")
print("=" * 70)
print(f"Overall R²: {df_stacking['r2'].mean():.4f}")
print(f"Overall MAE: {df_stacking['mae'].mean():.3f}")
print(f"\nBy zone:")
for zone in zones:
    zone_data = df_stacking[df_stacking['zone'] == zone]
    print(f"  {zone}: R²={zone_data['r2'].values[0]:.4f}, MAE={zone_data['mae'].values[0]:.3f}")

# Compare to tuned LightGBM baseline
print(f"\n" + "=" * 70)
print("COMPARISON: Stacking vs Tuned LightGBM")
print("=" * 70)
baseline_r2 = 0.341  # From tuned LightGBM
improvement = df_stacking['r2'].mean() - baseline_r2
print(f"Baseline (Tuned LightGBM) R²: {baseline_r2:.4f}")
print(f"Stacking R²: {df_stacking['r2'].mean():.4f}")
print(f"Improvement: {improvement:+.4f} ({improvement/baseline_r2*100:+.1f}%)")

print("\n✓ Stacking complete! Results saved.")
