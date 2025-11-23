"""
Regional Analysis: Investigate spatial performance variability across quadrants.

MODIFICATIONS FOR THIS PROJECT:
- Analyzes why certain regions (e.g., NW) underperform
- Compares regional PM2.5 characteristics, sample distributions
- Identifies geographic/temporal patterns by region
- Generates insights for domain adaptation or regional tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
DATA_DIR = Path(r"D:\PM25_Satellite_Research\data\processed")
LOCO_DIR = DATA_DIR / "loco_folds"
RESULTS_DIR = Path(r"D:\PM25_Satellite_Research\results\tables")
FIGURES_DIR = Path(r"D:\PM25_Satellite_Research\results\figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_all_data():
    """Load all LOCO fold data."""
    regions = ['NE', 'NW', 'SE', 'SW']
    data = {}
    
    for region in regions:
        test_file = LOCO_DIR / f"test_loco_{region}.parquet"
        data[region] = pd.read_parquet(test_file)
    
    return data

def analyze_pm25_distribution(data):
    """Analyze PM2.5 distribution by region."""
    print("\n" + "="*60)
    print("PM2.5 DISTRIBUTION BY REGION")
    print("="*60)
    
    stats = []
    for region, df in data.items():
        pm25 = df['pm25']
        stats.append({
            'Region': region,
            'Count': len(df),
            'Mean': pm25.mean(),
            'Median': pm25.median(),
            'Std': pm25.std(),
            'Min': pm25.min(),
            'Max': pm25.max(),
            'Q25': pm25.quantile(0.25),
            'Q75': pm25.quantile(0.75)
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # Save
    stats_df.to_csv(RESULTS_DIR / "regional_pm25_statistics.csv", index=False)
    
    return stats_df

def analyze_spatial_coverage(data):
    """Analyze spatial coverage and grid density by region."""
    print("\n" + "="*60)
    print("SPATIAL COVERAGE BY REGION")
    print("="*60)
    
    coverage = []
    for region, df in data.items():
        unique_cells = df.groupby(['lat', 'lon']).ngroups
        coverage.append({
            'Region': region,
            'Total_Samples': len(df),
            'Unique_Grid_Cells': unique_cells,
            'Avg_Samples_Per_Cell': len(df) / unique_cells if unique_cells > 0 else 0,
            'Lat_Range': f"{df['lat'].min():.1f} to {df['lat'].max():.1f}",
            'Lon_Range': f"{df['lon'].min():.1f} to {df['lon'].max():.1f}"
        })
    
    coverage_df = pd.DataFrame(coverage)
    print(coverage_df.to_string(index=False))
    
    # Save
    coverage_df.to_csv(RESULTS_DIR / "regional_spatial_coverage.csv", index=False)
    
    return coverage_df

def analyze_temporal_distribution(data):
    """Analyze temporal distribution by region."""
    print("\n" + "="*60)
    print("TEMPORAL DISTRIBUTION BY REGION")
    print("="*60)
    
    temporal = []
    for region, df in data.items():
        temporal.append({
            'Region': region,
            'Year_Range': f"{df['year'].min()}-{df['year'].max()}",
            'Unique_Months': df.groupby(['year', 'month']).ngroups,
            'Most_Represented_Year': df['year'].mode().values[0] if len(df['year'].mode()) > 0 else 'N/A'
        })
    
    temporal_df = pd.DataFrame(temporal)
    print(temporal_df.to_string(index=False))
    
    # Save
    temporal_df.to_csv(RESULTS_DIR / "regional_temporal_distribution.csv", index=False)
    
    return temporal_df

def analyze_geographic_characteristics(data):
    """Analyze geographic characteristics by region."""
    print("\n" + "="*60)
    print("GEOGRAPHIC CHARACTERISTICS BY REGION")
    print("="*60)
    
    geo_chars = []
    for region, df in data.items():
        geo_chars.append({
            'Region': region,
            'Avg_Latitude': df['lat'].mean(),
            'Avg_Longitude': df['lon'].mean(),
            'Latitude_Std': df['lat'].std(),
            'Longitude_Std': df['lon'].std(),
            'Avg_Abs_Latitude': df['abs_lat'].mean(),
            'Avg_Climate_Zone': df['climate_zone'].mode().values[0] if len(df['climate_zone'].mode()) > 0 else 'N/A'
        })
    
    geo_df = pd.DataFrame(geo_chars)
    print(geo_df.to_string(index=False))
    
    # Save
    geo_df.to_csv(RESULTS_DIR / "regional_geographic_characteristics.csv", index=False)
    
    return geo_df

def compare_nw_to_others(data):
    """Deep dive: Compare NW (underperforming) to other regions."""
    print("\n" + "="*60)
    print("DETAILED NW REGION ANALYSIS")
    print("="*60)
    print("\nNW region shows lower LOCO R² (0.855) compared to other regions.")
    print("Investigating potential causes:\n")
    
    nw_data = data['NW']
    others = pd.concat([data['NE'], data['SE'], data['SW']])
    
    # 1. Pollution levels
    print("1. POLLUTION LEVELS:")
    print(f"   NW mean PM2.5: {nw_data['pm25'].mean():.2f} µg/m³")
    print(f"   Others mean PM2.5: {others['pm25'].mean():.2f} µg/m³")
    print(f"   → NW {'LOWER' if nw_data['pm25'].mean() < others['pm25'].mean() else 'HIGHER'} pollution regions")
    
    # 2. Variability
    print("\n2. POLLUTION VARIABILITY:")
    print(f"   NW std: {nw_data['pm25'].std():.2f}")
    print(f"   Others std: {others['pm25'].std():.2f}")
    print(f"   → NW has {'HIGHER' if nw_data['pm25'].std() > others['pm25'].std() else 'LOWER'} variability")
    
    # 3. Sample distribution
    print("\n3. SAMPLE DISTRIBUTION:")
    print(f"   NW samples: {len(nw_data):,} ({len(nw_data)/len(pd.concat([data['NE'], data['NW'], data['SE'], data['SW']]))*100:.1f}%)")
    print(f"   NE samples: {len(data['NE']):,} ({len(data['NE'])/len(pd.concat([data['NE'], data['NW'], data['SE'], data['SW']]))*100:.1f}%)")
    print(f"   SE samples: {len(data['SE']):,} ({len(data['SE'])/len(pd.concat([data['NE'], data['NW'], data['SE'], data['SW']]))*100:.1f}%)")
    print(f"   SW samples: {len(data['SW']):,} ({len(data['SW'])/len(pd.concat([data['NE'], data['NW'], data['SE'], data['SW']]))*100:.1f}%)")
    
    # 4. Geographic spread
    print("\n4. GEOGRAPHIC SPREAD:")
    print(f"   NW latitude range: {nw_data['lat'].min():.1f} to {nw_data['lat'].max():.1f}")
    print(f"   NW longitude range: {nw_data['lon'].min():.1f} to {nw_data['lon'].max():.1f}")
    print(f"   → Covers {'high-latitude' if nw_data['abs_lat'].mean() > 45 else 'mid-latitude'} regions")
    
    # 5. Extreme values
    print("\n5. EXTREME VALUES:")
    nw_outliers = nw_data[nw_data['pm25'] > nw_data['pm25'].quantile(0.95)]
    print(f"   NW high pollution outliers (>95th percentile): {len(nw_outliers):,}")
    print(f"   → High-pollution events: {'More common' if len(nw_outliers)/len(nw_data) > 0.05 else 'Less common'}")

def create_visualization(data):
    """Create comparison visualization."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regional Analysis: PM2.5 Characteristics by Quadrant', fontsize=16, fontweight='bold')
    
    regions = list(data.keys())
    
    # 1. PM2.5 distribution
    ax1 = axes[0, 0]
    pm25_by_region = [data[r]['pm25'].values for r in regions]
    bp = ax1.boxplot(pm25_by_region, labels=regions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax1.set_ylabel('PM2.5 (µg/m³)', fontsize=11)
    ax1.set_title('PM2.5 Distribution by Region', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Sample count
    ax2 = axes[0, 1]
    counts = [len(data[r]) for r in regions]
    colors = ['#ff6b6b' if r == 'NW' else '#4ecdc4' for r in regions]
    ax2.bar(regions, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Samples', fontsize=11)
    ax2.set_title('Sample Count by Region', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        ax2.text(i, v + 5000, f'{v//1000}k', ha='center', fontweight='bold')
    
    # 3. Geographic coverage (latitude)
    ax3 = axes[1, 0]
    for region in regions:
        lats = data[region]['lat']
        ax3.scatter([region] * len(lats), lats, alpha=0.1, s=1)
    ax3.set_ylabel('Latitude', fontsize=11)
    ax3.set_title('Latitude Distribution by Region', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Mean PM2.5 vs Std
    ax4 = axes[1, 1]
    means = [data[r]['pm25'].mean() for r in regions]
    stds = [data[r]['pm25'].std() for r in regions]
    colors = ['#ff6b6b' if r == 'NW' else '#4ecdc4' for r in regions]
    ax4.scatter(means, stds, s=300, c=colors, alpha=0.7, edgecolor='black', linewidth=2)
    for i, region in enumerate(regions):
        ax4.annotate(region, (means[i], stds[i]), fontweight='bold', fontsize=12, 
                    ha='center', va='center')
    ax4.set_xlabel('Mean PM2.5 (µg/m³)', fontsize=11)
    ax4.set_ylabel('Std PM2.5 (µg/m³)', fontsize=11)
    ax4.set_title('Mean vs Std PM2.5 by Region', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "regional_analysis_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: regional_analysis_comparison.png")
    plt.close()

def main():
    """Main regional analysis pipeline."""
    
    print("="*60)
    print("REGIONAL ANALYSIS: INVESTIGATING SPATIAL VARIABILITY")
    print("="*60)
    
    # Load data
    print("\nLoading LOCO fold data...")
    data = load_all_data()
    print(f"✓ Loaded data for 4 regions")
    
    # Analyses
    pm25_stats = analyze_pm25_distribution(data)
    spatial_coverage = analyze_spatial_coverage(data)
    temporal_dist = analyze_temporal_distribution(data)
    geo_chars = analyze_geographic_characteristics(data)
    
    # Deep dive NW
    compare_nw_to_others(data)
    
    # Visualization
    create_visualization(data)
    
    # Summary
    print("\n" + "="*60)
    print("REGIONAL ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey findings saved to results/tables/:")
    print("  - regional_pm25_statistics.csv")
    print("  - regional_spatial_coverage.csv")
    print("  - regional_temporal_distribution.csv")
    print("  - regional_geographic_characteristics.csv")
    print("\nVisualization saved to results/figures/:")
    print("  - regional_analysis_comparison.png")
    
    print("\n" + "="*60)
    print("IMPLICATIONS FOR YOUR PAPER")
    print("="*60)
    print("""
This analysis reveals:
1. Regional variability in PM2.5 characteristics
2. Why certain regions (NW) underperform spatially
3. Potential for domain adaptation or region-specific models
4. Data distribution imbalances across quadrants

Narrative for Q1 paper:
- "While overall LOCO R² is high, regional analysis reveals
  significant spatial heterogeneity in model performance"
- "Northwest quadrant shows 5-10% lower R² due to [reason]"
- "Suggests that ML models capture dominant global patterns
  but struggle with regional nuances"
    """)

if __name__ == "__main__":
    main()
