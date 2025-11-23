import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\modeling")

def plot_shap_by_zone():
    fpath = RESULTS_DIR / "shap_by_zone.csv"
    try:
        df = pd.read_csv(fpath)
        categories = ['Geographic', 'Meteorology', 'Temporal']
        fig, ax = plt.subplots(figsize=(8,5))
        df[categories].plot(kind='bar', ax=ax, color=['#c44e52','#4c72b0','#55a868'], edgecolor='black')
        ax.set_xticklabels(df['Zone'], rotation=45)
        ax.set_ylabel('Mean |SHAP|')
        ax.set_title('SHAP Feature Category Importance by Köppen Zone')
        ax.legend(title='Category')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_by_zone_comparison.png", dpi=300)
        plt.close()
        print("✓ Saved: shap_by_zone_comparison.png")
    except Exception as e:
        print(f"[ERROR shap_by_zone]: {e}")

def plot_shap_heatmap():
    fpath = RESULTS_DIR / "geographic_dominance_heatmap.csv"
    try:
        df = pd.read_csv(fpath, index_col=0)
        plt.figure(figsize=(10,7))
        sns.heatmap(df, annot=True, fmt=".2f", cmap='Reds', cbar_kws={'label': 'Mean |SHAP|'})
        plt.title('SHAP Feature Importance Across Zones')
        plt.ylabel('Feature')
        plt.xlabel('Zone')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "geographic_dominance_heatmap.png", dpi=300)
        plt.close()
        print("✓ Saved: geographic_dominance_heatmap.png")
    except Exception as e:
        print(f"[ERROR shap_heatmap]: {e}")

def plot_temporal_vs_loco():
    fpath = RESULTS_DIR / "temporal_vs_loco_validation.csv"
    try:
        df = pd.read_csv(fpath)
        metrics = df['Metric']
        x = range(len(metrics))
        plt.figure(figsize=(7,5))
        plt.bar([i - 0.15 for i in x], df['LOCO'], width=0.3, label="LOCO", color="#4c72b0")
        plt.bar([i + 0.15 for i in x], df['Temporal'], width=0.3, label="Temporal", color="#c44e52")
        plt.xticks(x, metrics)
        plt.ylabel("Score / Value")
        plt.title("LOCO vs. Temporal Validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "temporal_vs_loco_validation.png", dpi=300)
        plt.close()
        print("✓ Saved: temporal_vs_loco_validation.png")
    except Exception as e:
        print(f"[ERROR temporal_vs_loco]: {e}")

def plot_tuning_impact():
    fpath = RESULTS_DIR / "tuning_impact_visualization.csv"
    try:
        df = pd.read_csv(fpath)
        labels = df['Metric']
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(7,5))
        ax.bar([i - 0.15 for i in x], df['Baseline'], width=0.3, color='#4c72b0', label='Baseline')
        ax.bar([i + 0.15 for i in x], df['Tuned'], width=0.3, color='#55a868', label='Tuned')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score / Value")
        ax.set_title("Model Performance: Baseline vs. Hyperparameter Tuning")
        ax.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tuning_impact_visualization.png", dpi=300)
        plt.close()
        print("✓ Saved: tuning_impact_visualization.png")
    except Exception as e:
        print(f"[ERROR tuning_impact]: {e}")

def plot_ensemble_stacking():
    fpath = RESULTS_DIR / "stacking_results.csv"
    try:
        df = pd.read_csv(fpath)
        # Replace with correct column names if different in your CSV!
        x = ['LightGBM (tuned)', 'Ensemble Stacking']
        y = [df['LightGBM_R2'].iloc[0], df['Stacking_R2'].iloc[0]]
        plt.figure(figsize=(6,5))
        plt.bar(x, y, color=['#4c72b0', '#c44e52'])
        plt.ylabel('R²')
        plt.title('Ensemble Stacking Degrades Performance')
        for i, v in enumerate(y):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "ensemble_stacking_failure.png", dpi=300)
        plt.close()
        print("✓ Saved: ensemble_stacking_failure.png")
    except Exception as e:
        print(f"[ERROR ensemble_stacking]: {e}")

# === Entry point to batch all plots ===
if __name__ == "__main__":
    print("GENERATING ALL PM2.5 FIGURES...")
    plot_shap_by_zone()
    plot_shap_heatmap()
    plot_temporal_vs_loco()
    plot_tuning_impact()
    plot_ensemble_stacking()
    print("All requested figures generated (if data available).")
