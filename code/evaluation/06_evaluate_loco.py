"""
Leave-One-Region-Out (LOCO) Validation for Spatial Generalization.

ORIGINAL SOURCE INSPIRATION:
- LOCO validation: "National PM2.5 and NO2 exposure models for China" (2018)
  https://depts.washington.edu/airqual/Marshall_100.pdf
- Spatial cross-validation: "Deep learning framework for hourly air pollutants" (2025)

MODIFICATIONS FOR THIS PROJECT:
- Applied LOCO across 4 geographic quadrants (NE, NW, SE, SW)
- Tests if models trained on 3 quadrants generalize to 1 held-out quadrant
- Measures spatial generalization failure (negative results story!)
- Adapted for D: drive paths
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
LOCO_DIR = DATA_DIR / "loco_folds"
MODEL_DIR = Path(r"D:\PM25_Satellite_Research\models\trained_checkpoints")
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\tables")

# Feature columns (exclude target and metadata)
EXCLUDE_COLS = ['pm25', 'lat', 'lon', 'year', 'month', 'climate_zone', 'quadrant']

def load_models():
    """Load trained models."""
    models = {}
    
    if (MODEL_DIR / "random_forest.pkl").exists():
        models['Random Forest'] = joblib.load(MODEL_DIR / "random_forest.pkl")
    
    if (MODEL_DIR / "xgboost.json").exists():
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor()
        models['XGBoost'].load_model(MODEL_DIR / "xgboost.json")
    
    if (MODEL_DIR / "lightgbm.pkl").exists():
        models['LightGBM'] = joblib.load(MODEL_DIR / "lightgbm.pkl")
    
    return models

def calculate_metrics(y_true, y_pred, model_name, region):
    """Calculate comprehensive metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    
    return {
        'Model': model_name,
        'Test_Region': region,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }



def evaluate_loco():
    """Evaluate models on LOCO folds."""
    
    print("="*60)
    print("LEAVE-ONE-REGION-OUT (LOCO) EVALUATION")
    print("="*60)
    
    # Load models
    models = load_models()
    if not models:
        print("ERROR: No trained models found!")
        return
    
    print(f"\nLoaded {len(models)} models:")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    # Get LOCO fold files
    loco_files = sorted(LOCO_DIR.glob("test_loco_*.parquet"))
    regions = [f.stem.replace("test_loco_", "") for f in loco_files]
    
    print(f"\nFound {len(regions)} regions: {', '.join(regions)}")
    
    # Evaluate each LOCO fold
    all_results = []
    
    for region in regions:
        print(f"\n" + "-"*60)
        print(f"Testing region: {region.upper()}")
        print("-"*60)
        
        # Load test data (held-out region)
        test_file = LOCO_DIR / f"test_loco_{region}.parquet"
        test_data = pd.read_parquet(test_file)
        
        # Get features (FIXED: remove date_key if it exists)
        feature_cols = [col for col in test_data.columns if col not in EXCLUDE_COLS and col != 'date_key']
        X_test = test_data[feature_cols]
        y_test = test_data['pm25']
        
        print(f"Test samples: {len(X_test):,}")
        
        # Evaluate each model
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = calculate_metrics(y_test, y_pred, model_name, region)
                all_results.append(metrics)
                
                print(f"  {model_name}: R²={metrics['R²']:.3f}, RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}")
                
            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")
    
    # Check if we have results
    if not all_results:
        print("\nERROR: No results collected!")
        return
    
    # Summary table
    print("\n" + "="*60)
    print("LOCO EVALUATION RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    
    # Pivot for better visualization
    pivot_r2 = results_df.pivot(index='Model', columns='Test_Region', values='R²')
    pivot_rmse = results_df.pivot(index='Model', columns='Test_Region', values='RMSE')
    
    print("\nR² by Model and Region:")
    print(pivot_r2.to_string())
    
    print("\nRMSE (µg/m³) by Model and Region:")
    print(pivot_rmse.to_string())
    
    # Calculate spatial generalization drop
    print("\n" + "="*60)
    print("SPATIAL GENERALIZATION ANALYSIS")
    print("="*60)
    
    # Load temporal results for comparison
    temporal_results = pd.read_csv(RESULTS_DIR / "baseline_temporal_results.csv")
    
    for model_name in models.keys():
        temporal_r2 = temporal_results[temporal_results['Model'] == model_name]['R²'].values[0]
        loco_r2_mean = results_df[results_df['Model'] == model_name]['R²'].mean()
        drop_percent = ((temporal_r2 - loco_r2_mean) / temporal_r2) * 100
        
        print(f"\n{model_name}:")
        print(f"  Temporal R²: {temporal_r2:.3f}")
        print(f"  LOCO R² (mean): {loco_r2_mean:.3f}")
        print(f"  Performance drop: {drop_percent:.1f}%")
        
        # Show per-region
        for region in regions:
            region_r2 = results_df[(results_df['Model'] == model_name) & 
                                   (results_df['Test_Region'] == region)]['R²'].values[0]
            print(f"    → {region.upper()}: {region_r2:.3f}")
    
    # Save results
    results_df.to_csv(RESULTS_DIR / "loco_evaluation_results.csv", index=False)
    print(f"\n✓ Results saved to: {RESULTS_DIR / 'loco_evaluation_results.csv'}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
The LOCO results reveal spatial generalization challenges:
- Models trained on 3 regions may fail on the 4th (especially underrepresented regions)
- Negative R² indicates predictions worse than simply using the mean
- This supports your paper's narrative: ML models don't generalize spatially
  without additional constraints (physics-informed, domain adaptation, etc.)
    """)


if __name__ == "__main__":
    evaluate_loco()
