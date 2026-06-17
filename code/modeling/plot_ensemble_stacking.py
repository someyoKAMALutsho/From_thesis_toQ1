from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).resolve().parent
CODE_DIR    = SCRIPT_DIR.parent
PROJECT_DIR = CODE_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "modeling"

# Load both CSVs
loco_csv = pd.read_csv(RESULTS_DIR / "loco_cv_results_gpu.csv")
stacking_csv = pd.read_csv(RESULTS_DIR / "stacking_results.csv")

# Extract zone names
zones = stacking_csv['zone']

# Get LightGBM R² by zone
lgb_rows = loco_csv[loco_csv['model'].str.lower().str.contains('lightgbm')]
lgb_r2 = lgb_rows.set_index('zone').loc[zones, 'r2']  # match stacking zone order

# Get stacking R² by zone
stacking_r2 = stacking_csv['r2']

# Plot
fig, ax = plt.subplots(figsize=(8,5))
bar_width = 0.35
x = range(len(zones))
ax.bar([i - bar_width/2 for i in x], lgb_r2, width=bar_width, label='LightGBM (Tuned)', color='#4c72b0')
ax.bar([i + bar_width/2 for i in x], stacking_r2, width=bar_width, label='Stacking (LGB+XGB+RF)', color='#c44e52')
ax.set_xticks(x)
ax.set_xticklabels(zones, rotation=45)
ax.set_ylabel('R²')
ax.set_title('R² by Zone: LightGBM vs Stacking Ensemble')
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'ensemble_stacking_failure.png', dpi=300)
plt.close()
print("✓ Saved: ensemble_stacking_failure.png")
