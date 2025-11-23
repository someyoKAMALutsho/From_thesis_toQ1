"""
GPU-Accelerated Baseline Model Training with Köppen LOCO Cross-Validation
For PM2.5 Satellite Prediction Paper (Q1 Journal)

CODE SOURCES & ATTRIBUTION:
===========================
1. XGBoost GPU Training:
   Source: https://xgboost.readthedocs.io/en/stable/gpu/index.html
   Reference: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD'16.
   ADAPTATION: tree_method='gpu_hist', device='cuda' for NVIDIA GTX 1650
   
2. LightGBM GPU Training:
   Source: https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html
   Reference: Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS'17.
   ADAPTATION: device='cuda' for GPU acceleration
   
3. Random Forest (scikit-learn):
   Source: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
   Reference: Breiman (2001). Random Forests. Machine Learning, 45(1), 5-32.
   
4. Linear Regression (scikit-learn):
   Source: https://scikit-learn.org/stable/modules/linear_model.html
   
5. Köppen LOCO Cross-Validation:
   Adapted from: Roberts et al. (2017). Cross-validation strategies for data with 
   temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913-929.
   ADAPTATION: Applied to climate zones instead of spatial blocks

HARDWARE: NVIDIA GeForce GTX 1650 (Compute Capability 7.5)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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

# Try to import GAM
try:
    from pygam import LinearGAM
    HAS_GAM = True
except ImportError:
    HAS_GAM = False
    print("⚠ pygam not available, GAM will be skipped")

# --- CONFIGURATION ---
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\modeling")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "pm25_merra2_meteorology_final.parquet"

# Features (prec excluded due to 100% missing in MERRA-2 monthly data)
FEATURE_COLS = [
    'lat', 'lon', 'year', 'month',
    't2m', 'ps', 'u10', 'v10', 'blh'
]
TARGET_COL = 'pm25'

print("=" * 70)
print("GPU-ACCELERATED BASELINE MODEL TRAINING")
print("=" * 70)

# Test GPU availability
print("\nTesting GPU availability...")
try:
    # Test XGBoost GPU
    test_X = np.random.rand(100, 10)
    test_y = np.random.rand(100)
    test_model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=5, verbosity=0)
    test_model.fit(test_X, test_y)
    print("✓ XGBoost GPU: Working")
    XGB_GPU = True
except Exception as e:
    print(f"⚠ XGBoost GPU failed: {e}")
    print("  Falling back to CPU")
    XGB_GPU = False

try:
    # Test LightGBM GPU
    test_lgb = lgb.LGBMRegressor(device='cuda', n_estimators=5, verbose=-1)
    test_lgb.fit(test_X, test_y)
    print("✓ LightGBM GPU: Working")
    LGB_GPU = True
except Exception as e:
    print(f"⚠ LightGBM GPU failed: {e}")
    print("  Falling back to CPU")
    LGB_GPU = False

# Load dataset
print("\nLoading dataset...")
df_full = pd.read_parquet(INPUT_FILE)
print(f"Total samples: {len(df_full):,}")
print(f"Features: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")

# Get Köppen zones
zones = sorted(df_full['koppen_zone'].unique())
print(f"Köppen zones: {zones}")

# Results storage
results = {
    'zone': [],
    'model': [],
    'train_samples': [],
    'test_samples': [],
    'mae': [],
    'rmse': [],
    'r2': [],
    'bias': [],
    'explained_variance': [],
    'train_time_sec': []
}

# ============================================================================
# LOCO CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING WITH LOCO CROSS-VALIDATION")
print("=" * 70)

for test_zone in zones:
    print(f"\n{'='*70}")
    print(f"Testing on {test_zone} climate zone")
    print(f"{'='*70}")
    
    # Split data
    train_data = df_full[df_full['koppen_zone'] != test_zone].copy()
    test_data = df_full[df_full['koppen_zone'] == test_zone].copy()
    
    print(f"Train: {len(train_data):,} samples | Test: {len(test_data):,} samples")
    
    # Prepare features and target
    X_train = train_data[FEATURE_COLS].copy()
    y_train = train_data[TARGET_COL].copy()
    X_test = test_data[FEATURE_COLS].copy()
    y_test = test_data[TARGET_COL].copy()
    
    # Imputation
    print("  Imputing missing values (median)...", end=" ", flush=True)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_train = pd.DataFrame(X_train_imputed, columns=FEATURE_COLS)
    X_test = pd.DataFrame(X_test_imputed, columns=FEATURE_COLS)
    print("✓")
    
    # Derive wind_speed
    X_train['wind_speed'] = np.sqrt(X_train['u10']**2 + X_train['v10']**2)
    X_test['wind_speed'] = np.sqrt(X_test['u10']**2 + X_test['v10']**2)
    
    ALL_FEATURES = FEATURE_COLS + ['wind_speed']
    
    # Standardize for Linear Regression and GAM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[ALL_FEATURES])
    X_test_scaled = scaler.transform(X_test[ALL_FEATURES])
    
    # ===== Model 1: XGBoost =====
    device_label = "GPU" if XGB_GPU else "CPU"
    print(f"  Training XGBoost ({device_label})...", end=" ", flush=True)
    t0 = time.time()
    
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    if XGB_GPU:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['device'] = 'cuda'
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train[ALL_FEATURES], y_train)
    y_pred_xgb = xgb_model.predict(X_test[ALL_FEATURES])
    t_xgb = time.time() - t0
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    bias_xgb = np.mean(y_pred_xgb - y_test)
    print(f"✓ MAE={mae_xgb:.3f}, R²={r2_xgb:.3f} ({t_xgb:.1f}s)")
    
    results['zone'].append(test_zone)
    results['model'].append(f'XGBoost-{device_label}')
    results['train_samples'].append(len(train_data))
    results['test_samples'].append(len(test_data))
    results['mae'].append(mae_xgb)
    results['rmse'].append(rmse_xgb)
    results['r2'].append(r2_xgb)
    results['bias'].append(bias_xgb)
    results['explained_variance'].append(1 - (np.var(y_test - y_pred_xgb) / np.var(y_test)))
    results['train_time_sec'].append(t_xgb)
    
    # ===== Model 2: LightGBM =====
    device_label = "GPU" if LGB_GPU else "CPU"
    print(f"  Training LightGBM ({device_label})...", end=" ", flush=True)
    t0 = time.time()
    
  
    
    
    
    
    lgb_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.15,
    'subsample': 0.6,
    'colsample_bytree': 1.0,
    'num_leaves': 70,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1
                 }

    
    if LGB_GPU:
        lgb_params['device'] = 'cuda'
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train[ALL_FEATURES], y_train)
    y_pred_lgb = lgb_model.predict(X_test[ALL_FEATURES])
    t_lgb = time.time() - t0
    
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)
    bias_lgb = np.mean(y_pred_lgb - y_test)
    print(f"✓ MAE={mae_lgb:.3f}, R²={r2_lgb:.3f} ({t_lgb:.1f}s)")
    
    results['zone'].append(test_zone)
    results['model'].append(f'LightGBM-{device_label}')
    results['train_samples'].append(len(train_data))
    results['test_samples'].append(len(test_data))
    results['mae'].append(mae_lgb)
    results['rmse'].append(rmse_lgb)
    results['r2'].append(r2_lgb)
    results['bias'].append(bias_lgb)
    results['explained_variance'].append(1 - (np.var(y_test - y_pred_lgb) / np.var(y_test)))
    results['train_time_sec'].append(t_lgb)
    
    # ===== Model 3: Random Forest =====
    print(f"  Training Random Forest (CPU)...", end=" ", flush=True)
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train[ALL_FEATURES], y_train)
    y_pred_rf = rf_model.predict(X_test[ALL_FEATURES])
    t_rf = time.time() - t0
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    bias_rf = np.mean(y_pred_rf - y_test)
    print(f"✓ MAE={mae_rf:.3f}, R²={r2_rf:.3f} ({t_rf:.1f}s)")
    
    results['zone'].append(test_zone)
    results['model'].append('Random Forest-CPU')
    results['train_samples'].append(len(train_data))
    results['test_samples'].append(len(test_data))
    results['mae'].append(mae_rf)
    results['rmse'].append(rmse_rf)
    results['r2'].append(r2_rf)
    results['bias'].append(bias_rf)
    results['explained_variance'].append(1 - (np.var(y_test - y_pred_rf) / np.var(y_test)))
    results['train_time_sec'].append(t_rf)
    
    # ===== Model 4: Linear Regression =====
    print(f"  Training Linear Regression (CPU)...", end=" ", flush=True)
    t0 = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    t_lr = time.time() - t0
    
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    bias_lr = np.mean(y_pred_lr - y_test)
    print(f"✓ MAE={mae_lr:.3f}, R²={r2_lr:.3f} ({t_lr:.1f}s)")
    
    results['zone'].append(test_zone)
    results['model'].append('Linear Regression-CPU')
    results['train_samples'].append(len(train_data))
    results['test_samples'].append(len(test_data))
    results['mae'].append(mae_lr)
    results['rmse'].append(rmse_lr)
    results['r2'].append(r2_lr)
    results['bias'].append(bias_lr)
    results['explained_variance'].append(1 - (np.var(y_test - y_pred_lr) / np.var(y_test)))
    results['train_time_sec'].append(t_lr)
    
    # ===== Model 5: GAM =====
    if HAS_GAM:
        print(f"  Training GAM (CPU)...", end=" ", flush=True)
        t0 = time.time()
        gam_model = LinearGAM()
        gam_model.fit(X_train_scaled, y_train)
        y_pred_gam = gam_model.predict(X_test_scaled)
        t_gam = time.time() - t0
        
        mae_gam = mean_absolute_error(y_test, y_pred_gam)
        rmse_gam = np.sqrt(mean_squared_error(y_test, y_pred_gam))
        r2_gam = r2_score(y_test, y_pred_gam)
        bias_gam = np.mean(y_pred_gam - y_test)
        print(f"✓ MAE={mae_gam:.3f}, R²={r2_gam:.3f} ({t_gam:.1f}s)")
        
        results['zone'].append(test_zone)
        results['model'].append('GAM-CPU')
        results['train_samples'].append(len(train_data))
        results['test_samples'].append(len(test_data))
        results['mae'].append(mae_gam)
        results['rmse'].append(rmse_gam)
        results['r2'].append(r2_gam)
        results['bias'].append(bias_gam)
        results['explained_variance'].append(1 - (np.var(y_test - y_pred_gam) / np.var(y_test)))
        results['train_time_sec'].append(t_gam)

# ============================================================================
# SAVE RESULTS
# ============================================================================

df_results = pd.DataFrame(results)
results_csv = RESULTS_DIR / "loco_cv_results_gpu.csv"
df_results.to_csv(results_csv, index=False)
print(f"\n✓ Results saved: {results_csv}")

# Summary
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY (Mean Across All Zones)")
print("=" * 70)
summary = df_results.groupby('model')[['mae', 'rmse', 'r2', 'train_time_sec']].mean()
print(summary.round(4))

print("\nPerformance by Köppen Zone:")
zone_summary = df_results.groupby('zone')[['mae', 'rmse', 'r2']].mean()
print(zone_summary.round(4))

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("\n" + "=" * 70)
print("Generating Publication Figures...")
print("=" * 70)

# Plot 1: Performance by Zone
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pivot_r2 = df_results.pivot(index='zone', columns='model', values='r2')
pivot_r2.plot(kind='bar', ax=axes[0])
axes[0].set_title('R² Score by Köppen Zone', fontsize=12, fontweight='bold')
axes[0].set_ylabel('R² Score')
axes[0].set_xlabel('Köppen Climate Zone')
axes[0].legend(title='Model', fontsize=8, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

pivot_mae = df_results.pivot(index='zone', columns='model', values='mae')
pivot_mae.plot(kind='bar', ax=axes[1])
axes[1].set_title('MAE by Köppen Zone', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE (µg/m³)')
axes[1].set_xlabel('Köppen Climate Zone')
axes[1].legend(title='Model', fontsize=8, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "performance_by_zone_gpu.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: performance_by_zone_gpu.png")
plt.close()

# Plot 2: Model Comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_avg = df_results.groupby('model')[['mae', 'rmse']].mean()
model_avg.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Model Comparison: MAE and RMSE', fontsize=12, fontweight='bold')
ax.set_ylabel('Error (µg/m³)')
ax.set_xlabel('Model')
ax.legend(['MAE', 'RMSE'])
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison_gpu.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: model_comparison_gpu.png")
plt.close()

# Plot 3: R² Comparison
fig, ax = plt.subplots(figsize=(10, 6))
model_r2 = df_results.groupby('model')['r2'].mean().sort_values(ascending=False)
model_r2.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Mean R² Score by Model', fontsize=12, fontweight='bold')
ax.set_xlabel('R² Score')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "r2_by_model_gpu.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: r2_by_model_gpu.png")
plt.close()

print("\n" + "=" * 70)
print("✓ GPU-ACCELERATED TRAINING COMPLETE!")
print("=" * 70)
print(f"\nResults directory: {RESULTS_DIR}/")
print(f"  - loco_cv_results_gpu.csv")
print(f"  - performance_by_zone_gpu.png")
print(f"  - model_comparison_gpu.png")
print(f"  - r2_by_model_gpu.png")
print(f"\n🚀 Ready for paper writing!")
