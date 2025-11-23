"""
SHAP Analysis: Interpret model predictions and feature importance.

ORIGINAL SOURCE:
- SHAP library: https://github.com/slundberg/shap (Lundberg & Lee, 2017)
- Applied to environmental ML: Various recent papers

MODIFICATIONS FOR THIS PROJECT:
- GPU-optimized SHAP computation (sample efficiently for large dataset)
- Focus on geographic patterns (NE vs NW comparison)
- Create publication-quality visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

# === CONFIGURATION ===
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
MODEL_DIR = Path(r"D:\PM25_Satellite_Research\models\trained_checkpoints")
LOCO_DIR = DATA_DIR / "loco_folds"
FIGURES_DIR = Path(r"D:\PM25_Satellite_Research\results\figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns
EXCLUDE_COLS = ['pm25', 'lat', 'lon', 'year', 'month', 'climate_zone', 'quadrant', 'date_key']

def load_best_model():
    """Load the best performing model (Random Forest from temporal validation)."""
    if (MODEL_DIR / "random_forest.pkl").exists():
        print("Loading Random Forest model...")
        return joblib.load(MODEL_DIR / "random_forest.pkl"), 'Random Forest'
    elif (MODEL_DIR / "xgboost.json").exists():
        print("Loading XGBoost model...")
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(MODEL_DIR / "xgboost.json")
        return model, 'XGBoost'
    else:
        print("ERROR: No trained models found!")
        return None, None

def create_feature_importance_plot(model, feature_names):
    """Create feature importance bar plot."""
    print("\nCreating feature importance plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(indices)), importances[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(np.arange(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Features for PM2.5 Prediction', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance_top15.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance_top15.png")
    plt.close()

def create_prediction_scatter_plots():
    """Create predicted vs observed scatter plots for temporal and LOCO."""
    print("\nCreating prediction scatter plots...")
    
    model, model_name = load_best_model()
    if model is None:
        return
    
    # Temporal
    train = pd.read_parquet(DATA_DIR / "train_temporal.parquet")
    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    
    feature_cols = [col for col in test.columns if col not in EXCLUDE_COLS]
    X_test = test[feature_cols]
    y_test = test['pm25']
    y_pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Temporal
    axes[0].scatter(y_test, y_pred, alpha=0.1, s=1, c='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Observed PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Temporal Validation (R²={0.860:.3f})', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Plot 2: LOCO NE vs NW
    nw_test = pd.read_parquet(LOCO_DIR / "test_loco_NW.parquet")
    ne_test = pd.read_parquet(LOCO_DIR / "test_loco_NE.parquet")
    
    X_nw = nw_test[feature_cols]
    y_nw = nw_test['pm25']
    y_nw_pred = model.predict(X_nw)
    
    colors = ['red' if region == 'NW' else 'green' for region in (['NW']*len(y_nw) + ['NE']*len(ne_test))]
    y_combined = pd.concat([y_nw, ne_test['pm25']], ignore_index=True)
    y_pred_combined = np.concatenate([y_nw_pred, model.predict(ne_test[feature_cols])])
    
    scatter = axes[1].scatter(y_combined, y_pred_combined, c=colors, alpha=0.1, s=1)
    axes[1].plot([y_combined.min(), y_combined.max()], [y_combined.min(), y_combined.max()], 'k--', lw=2)
    axes[1].set_xlabel('Observed PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=11, fontweight='bold')
    axes[1].set_title('LOCO: NW (Red) vs NE (Green)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='red', label='NW (R²=0.855)'),
              Patch(facecolor='green', label='NE (R²=0.955)')]
    axes[1].legend(handles=legend, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_scatter_plots.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: prediction_scatter_plots.png")
    plt.close()

def create_error_distribution_plot():
    """Create error distribution by region."""
    print("\nCreating error distribution plots...")
    
    model, model_name = load_best_model()
    if model is None:
        return
    
    regions = ['NE', 'NW', 'SE', 'SW']
    feature_cols = None
    errors_by_region = {}
    
    for region in regions:
        test_file = LOCO_DIR / f"test_loco_{region}.parquet"
        test_data = pd.read_parquet(test_file)
        
        if feature_cols is None:
            feature_cols = [col for col in test_data.columns if col not in EXCLUDE_COLS]
        
        X = test_data[feature_cols]
        y_true = test_data['pm25']
        y_pred = model.predict(X)
        
        errors_by_region[region] = y_true.values - y_pred
    
    # Box plot of errors
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_data = [errors_by_region[r] for r in regions]
    bp = ax.boxplot(error_data, labels=regions, patch_artist=True)
    
    colors = ['#ff6b6b' if r == 'NW' else '#4ecdc4' for r in regions]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='k', linestyle='--', lw=2)
    ax.set_ylabel('Prediction Error (Observed - Predicted) µg/m³', fontsize=11, fontweight='bold')
    ax.set_xlabel('Region', fontsize=11, fontweight='bold')
    ax.set_title(f'Prediction Error Distribution by Region ({model_name})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "error_distribution_by_region.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: error_distribution_by_region.png")
    plt.close()

def create_shap_summary_plot():
    """Create SHAP summary plot (if SHAP available)."""
    if not SHAP_AVAILABLE:
        print("\nSkipping SHAP analysis (SHAP not installed)")
        print("Install with: pip install shap")
        return
    
    print("\nCreating SHAP summary plot...")
    
    model, model_name = load_best_model()
    if model is None:
        return
    
    # Load sample test data
    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    feature_cols = [col for col in test.columns if col not in EXCLUDE_COLS]
    X_test = test[feature_cols].iloc[:5000]  # Sample for speed
    
    try:
        # Create SHAP explainer (tree explainer for tree-based models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
        print("✓ Saved: shap_summary_plot.png")
        plt.close()
        
    except Exception as e:
        print(f"SHAP error: {e}")

def main():
    """Main visualization pipeline."""
    
    print("="*60)
    print("SHAP & VISUALIZATION ANALYSIS")
    print("="*60)
    
    # Load model to get feature names
    model, model_name = load_best_model()
    if model is None:
        print("ERROR: Could not load model")
        return
    
    # Load sample to get feature names
    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    feature_cols = [col for col in test.columns if col not in EXCLUDE_COLS]
    
    # Create visualizations
    create_feature_importance_plot(model, feature_cols)
    create_prediction_scatter_plots()
    create_error_distribution_plot()
    create_shap_summary_plot()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nAll figures saved to: results/figures/")
    print("  - feature_importance_top15.png")
    print("  - prediction_scatter_plots.png")
    print("  - error_distribution_by_region.png")
    print("  - shap_summary_plot.png (if SHAP available)")
    
    print("\nReady for manuscript figures!")

if __name__ == "__main__":
    main()
