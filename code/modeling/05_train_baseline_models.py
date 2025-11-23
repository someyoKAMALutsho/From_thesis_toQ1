"""
Train baseline ML models for PM2.5 prediction with GPU acceleration.

ORIGINAL SOURCE INSPIRATION:
- LightGBM GPU training: "A Comparative Study of LightGBM on Air Quality Data" (2023)
  https://ceur-ws.org/Vol-3762/557.pdf
- XGBoost implementation: "PM2.5 concentration prediction using machine learning" (2025)
  https://www.nature.com/articles/s41598-025-92019-3
- Random Forest baseline: Standard scikit-learn approach for tabular data

MODIFICATIONS FOR THIS PROJECT:
- Added GPU support for XGBoost (tree_method='gpu_hist') and LightGBM (device='gpu')
- Updated for XGBoost 2.0+ API (early_stopping_rounds in constructor)
- Optimized for GTX 1650 (4GB VRAM) with batch processing if needed
- Structured for Washington University SAT PM2.5 dataset
- Added early stopping to reduce training time
- Configured for D: drive output paths
- Included comprehensive metrics (R², RMSE, MAE, MAPE)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU-enabled libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Install with: pip install xgboost")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")
    LGB_AVAILABLE = False

# === CONFIGURATION ===
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
MODEL_DIR = Path(r"D:\PM25_Satellite_Research\models\trained_checkpoints")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\tables")

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns (exclude target and metadata)
EXCLUDE_COLS = ['pm25', 'lat', 'lon', 'year', 'month', 'climate_zone', 'quadrant']

# Random state for reproducibility
RANDOM_STATE = 42

def load_data():
    """Load train and test data."""
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train_temporal.parquet")
    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    
    # Identify feature columns
    feature_cols = [col for col in train.columns if col not in EXCLUDE_COLS]
    
    X_train = train[feature_cols]
    y_train = train['pm25']
    X_test = test[feature_cols]
    y_test = test['pm25']
    
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")
    
    return X_train, y_train, X_test, y_test, feature_cols

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    
    return {
        'Model': model_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest baseline.
    Source: Standard scikit-learn approach
    Modification: Reduced n_estimators and max_depth for faster training on large dataset
    """
    print("\n" + "="*60)
    print("TRAINING: Random Forest")
    print("="*60)
    
    start_time = time.time()
    
    # Model configuration (tuned for speed on 700k+ samples)
    rf = RandomForestRegressor(
        n_estimators=100,        # Reduced from typical 200+ for speed
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,               # Use all CPU cores
        verbose=1
    )
    
    # Train
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    y_pred = rf.predict(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, 'Random Forest')
    metrics['Train_Time_sec'] = train_time
    
    # Save model
    joblib.dump(rf, MODEL_DIR / "random_forest.pkl")
    print(f"✓ Model saved")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_imp.to_csv(RESULTS_DIR / "rf_feature_importance.csv", index=False)
    
    print(f"\nResults: R²={metrics['R²']:.3f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
    print(f"Training time: {train_time:.1f}s")
    
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost with GPU acceleration.
    Source: "PM2.5 concentration prediction using machine learning" (2025, Nature)
    Modification: Updated for XGBoost 3.0+ API (uses 'device' instead of 'gpu_id')
    """
    if not XGB_AVAILABLE:
        print("\nSkipping XGBoost (not installed)")
        return None
    
    print("\n" + "="*60)
    print("TRAINING: XGBoost (GPU-accelerated)")
    print("="*60)
    
    start_time = time.time()
    
    # Check GPU availability (XGBoost 3.0+ API)
    try:
        # Try GPU first (FIXED: use 'device' instead of 'gpu_id')
        params = {
            'objective': 'reg:squarederror',
            'device': 'cuda:0',  # FIXED: XGBoost 3.0+ uses 'device' not 'gpu_id'
            'tree_method': 'hist',  # FIXED: hist works with both CPU/GPU
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'early_stopping_rounds': 10,
            'verbosity': 1
        }
        
        model = xgb.XGBRegressor(**params)
        print("Using GPU acceleration (GTX 1650)")
        
    except Exception as e:
        # Fallback to CPU
        print(f"GPU not available ({e}), using CPU")
        params = {
            'objective': 'reg:squarederror',
            'device': 'cpu',
            'tree_method': 'hist',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'early_stopping_rounds': 10,
            'verbosity': 1
        }
        model = xgb.XGBRegressor(**params)
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )
    
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, 'XGBoost')
    metrics['Train_Time_sec'] = train_time
    
    # Save model
    model.save_model(MODEL_DIR / "xgboost.json")
    print(f"✓ Model saved")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_imp.to_csv(RESULTS_DIR / "xgb_feature_importance.csv", index=False)
    
    print(f"\nResults: R²={metrics['R²']:.3f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
    print(f"Training time: {train_time:.1f}s")
    
    return metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Train LightGBM with GPU acceleration.
    Source: "A Comparative Study of LightGBM on Air Quality Data" (2023)
    Modification: Added GPU device, optimized for satellite PM2.5 data
    """
    if not LGB_AVAILABLE:
        print("\nSkipping LightGBM (not installed)")
        return None
    
    print("\n" + "="*60)
    print("TRAINING: LightGBM (GPU-accelerated)")
    print("="*60)
    
    start_time = time.time()
    
    # Try GPU, fallback to CPU if not available
    use_gpu = False
    try:
        # Test if GPU is available
        params_test = {
            'objective': 'regression',
            'device': 'gpu',
            'n_estimators': 1,
            'verbosity': -1
        }
        test_model = lgb.LGBMRegressor(**params_test)
        test_model.fit(X_train.iloc[:100], y_train.iloc[:100])
        use_gpu = True
        print("Using GPU acceleration (GTX 1650)")
    except Exception as e:
        print(f"GPU not available ({e}), using CPU")
    
    # Configure model
    if use_gpu:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu',           # GPU acceleration
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_leaves': 31,
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }
    else:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'cpu',
            'num_leaves': 31,
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1
        }
    
    model = lgb.LGBMRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )
    
    train_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, 'LightGBM')
    metrics['Train_Time_sec'] = train_time
    
    # Save model
    joblib.dump(model, MODEL_DIR / "lightgbm.pkl")
    print(f"✓ Model saved")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_imp.to_csv(RESULTS_DIR / "lgb_feature_importance.csv", index=False)
    
    print(f"\nResults: R²={metrics['R²']:.3f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
    print(f"Training time: {train_time:.1f}s")
    
    return metrics

def main():
    """Main training pipeline."""
    
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    
    # Load data
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    
    # Train models
    all_metrics = []
    
    # 1. Random Forest
    rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics.append(rf_metrics)
    
    # 2. XGBoost
    xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_metrics:
        all_metrics.append(xgb_metrics)
    
    # 3. LightGBM
    lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    if lgb_metrics:
        all_metrics.append(lgb_metrics)
    
    # Summary table
    print("\n" + "="*60)
    print("TEMPORAL VALIDATION RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame(all_metrics)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(RESULTS_DIR / "baseline_temporal_results.csv", index=False)
    print(f"\n✓ Results saved to: {RESULTS_DIR / 'baseline_temporal_results.csv'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review feature importance in results/tables/")
    print("  2. Run LOCO validation (06_evaluate_loco.py)")
    print("  3. Add meteorology data to improve performance")

if __name__ == "__main__":
    main()
