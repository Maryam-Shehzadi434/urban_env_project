"""
Task 3: Distribution Modeling & Tail Integrity
Urban Environmental Intelligence Challenge
Author: Data Architect
Date: 2025

Purpose: Analyze PM2.5 distribution for industrial zones to find extreme hazard events
         (>200 μg/m³) and determine the 99th percentile.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # This is the scipy stats module
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXTREME_THRESHOLD = 200  # μg/m³ - Extreme hazard events
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def get_file_path():
    """Get the correct path to the data file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'data', 'raw', 'wide_format.parquet'), script_dir

def load_and_prepare_data():
    """Load data and prepare for analysis"""
    print("="*80)
    print("TASK 3: DISTRIBUTION MODELING & TAIL INTEGRITY")
    print("="*80)
    
    filepath, script_dir = get_file_path()
    print(f"\n📂 Loading data from: {filepath}")
    
    df = pd.read_parquet(filepath)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df, script_dir

# =============================================================================
# STEP 1: SELECT INDUSTRIAL ZONE STATION
# =============================================================================

def select_industrial_zone(df):
    """
    Select an industrial zone station for detailed analysis
    Using the same zone classification logic from Task 1
    """
    print("\n" + "="*80)
    print("STEP 1: SELECTING INDUSTRIAL ZONE STATION")
    print("="*80)
    
    # Recreate zone labels using KMeans (same as Task 1)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Aggregate to station level
    pollution_vars = ['pm25', 'pm10', 'no2']
    station_df = df.groupby('station_id')[pollution_vars].mean().reset_index()
    
    # Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(station_df[pollution_vars])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    station_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Determine industrial cluster (higher pollution)
    cluster_means = station_df.groupby('cluster')[pollution_vars].mean()
    industrial_cluster = cluster_means.mean(axis=1).idxmax()
    
    # Get industrial station IDs
    industrial_stations = station_df[station_df['cluster'] == industrial_cluster]['station_id'].tolist()
    
    print(f"\n🏭 Found {len(industrial_stations)} industrial stations")
    
    # Select the station with highest PM2.5 (worst case)
    station_means = df[df['station_id'].isin(industrial_stations)].groupby('station_id')['pm25'].mean()
    selected_station = station_means.idxmax()
    max_pm25 = station_means.max()
    
    print(f"\n📌 Selected industrial station: {selected_station}")
    print(f"   • Mean PM2.5: {max_pm25:.2f} μg/m³")
    print(f"   • Total hours: {len(df[df['station_id'] == selected_station]):,}")
    
    # Extract data for this station
    station_data = df[df['station_id'] == selected_station]['pm25'].dropna().values
    
    return selected_station, station_data, industrial_stations

# =============================================================================
# STEP 2: BASIC STATISTICS AND EXTREME EVENT ANALYSIS
# =============================================================================

def calculate_basic_stats(data, selected_station):
    """Calculate basic statistics and extreme event probabilities"""
    print("\n" + "="*80)
    print("STEP 2: BASIC STATISTICS & EXTREME EVENT ANALYSIS")
    print("="*80)
    
    n_hours = len(data)
    n_extreme = np.sum(data > EXTREME_THRESHOLD)
    prob_extreme = n_extreme / n_hours if n_hours > 0 else 0
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]
    percentile_values = np.percentile(data, percentiles)
    
    print(f"\n📊 Station {selected_station} PM2.5 Statistics:")
    print(f"   • Total hours analyzed: {n_hours:,}")
    print(f"   • Mean: {np.mean(data):.2f} μg/m³")
    print(f"   • Median: {np.median(data):.2f} μg/m³")
    print(f"   • Std Dev: {np.std(data):.2f} μg/m³")
    print(f"   • Minimum: {np.min(data):.2f} μg/m³")
    print(f"   • Maximum: {np.max(data):.2f} μg/m³")
    
    print(f"\n⚠️ EXTREME HAZARD EVENTS (PM2.5 > {EXTREME_THRESHOLD}):")
    print(f"   • Number of events: {n_extreme}")
    print(f"   • Probability: {prob_extreme:.6f} ({prob_extreme*100:.4f}%)")
    if prob_extreme > 0:
        print(f"   • Return period: {1/prob_extreme:.0f} hours ({1/prob_extreme/24:.1f} days)")
    else:
        print(f"   • Return period: Never occurred")
    
    print("\n📈 PERCENTILES:")
    for p, val in zip(percentiles, percentile_values):
        print(f"   • {p}th percentile: {val:.2f} μg/m³")
    
    # Compare 99th percentile with threshold
    p99 = percentile_values[percentiles.index(99)]
    if p99 > EXTREME_THRESHOLD:
        print(f"\n🔴 99th percentile ({p99:.2f}) EXCEEDS extreme threshold ({EXTREME_THRESHOLD})")
        print(f"   → Extreme events occur MORE than 1% of the time")
    else:
        print(f"\n🟢 99th percentile ({p99:.2f}) is BELOW extreme threshold ({EXTREME_THRESHOLD})")
        print(f"   → Extreme events occur LESS than 1% of the time")
    
    # Return dictionary with statistics (RENAMED from 'stats' to 'statistics' to avoid conflict)
    statistics = {
        'n_hours': n_hours,
        'n_extreme': n_extreme,
        'prob_extreme': prob_extreme,
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'percentiles': dict(zip(percentiles, percentile_values))
    }
    
    return statistics

# =============================================================================
# STEP 3: PLOT 1 - OPTIMIZED FOR PEAKS
# =============================================================================

def create_peaks_plot(data, statistics, selected_station, output_dir):
    """
    Create distribution plot optimized to reveal PEAKS
    Focuses on the main body of the distribution (0-100 μg/m³)
    """
    print("\n" + "="*80)
    print("STEP 3: CREATING PLOT 1 - OPTIMIZED FOR PEAKS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== PLOT 1.1: Histogram with optimal bin width =====
    ax1 = axes[0, 0]
    
    # Use Freedman-Diaconis rule for optimal bin width
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    n_bins = int((np.max(data) - np.min(data)) / bin_width)
    n_bins = max(30, min(100, n_bins))  # Cap between 30 and 100
    
    ax1.hist(data, bins=n_bins, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add KDE (Kernel Density Estimate) - FIXED: using scipy.stats, not the dictionary
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(0, min(150, np.max(data)), 1000)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax1.axvline(statistics['mean'], color='darkred', linestyle='--', linewidth=2, 
                label=f"Mean: {statistics['mean']:.1f}")
    ax1.axvline(statistics['median'], color='darkgreen', linestyle='--', linewidth=2, 
                label=f"Median: {statistics['median']:.1f}")
    
    ax1.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Histogram with KDE (Optimal Bins)', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, min(150, np.max(data)))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== PLOT 1.2: Box Plot =====
    ax2 = axes[0, 1]
    
    # Create box plot with outliers
    bp = ax2.boxplot(data, vert=True, patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['fliers'][0].set_marker('o')
    bp['fliers'][0].set_markerfacecolor('red')
    bp['fliers'][0].set_markersize(4)
    bp['fliers'][0].set_alpha(0.5)
    
    ax2.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot with Outliers', fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for IQR
    q1, q3 = np.percentile(data, [25, 75])
    ax2.text(1.1, q1, f'Q1: {q1:.1f}', fontsize=9, va='center')
    ax2.text(1.1, q3, f'Q3: {q3:.1f}', fontsize=9, va='center')
    
    # ===== PLOT 1.3: Violin Plot =====
    ax3 = axes[1, 0]
    
    parts = ax3.violinplot(data, positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][0].set_edgecolor('black')
    
    ax3.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax3.set_title('Violin Plot (Shows Distribution Shape)', fontsize=13, fontweight='bold')
    ax3.set_xticks([])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ===== PLOT 1.4: Rug Plot + Density =====
    ax4 = axes[1, 1]
    
    # Plot density
    ax4.plot(x_range, kde(x_range), 'b-', linewidth=2, alpha=0.8)
    ax4.fill_between(x_range, kde(x_range), alpha=0.3, color='steelblue')
    
    # Add rug plot (small ticks for each data point - sample for visibility)
    sample_idx = np.random.choice(len(data), size=min(500, len(data)), replace=False)
    ax4.plot(data[sample_idx], np.zeros_like(data[sample_idx]), '|', color='red', markersize=10, alpha=0.3)
    
    ax4.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax4.set_title('Density Plot with Rug (Raw Data)', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, min(150, np.max(data)))
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'PLOT 1: DISTRIBUTIONS OPTIMIZED FOR PEAKS\nStation {selected_station} - Focus on Main Body (0-150 μg/m³)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task3_peaks_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Peaks plot saved to: {output_path}")
    plt.show()
    plt.close()

# =============================================================================
# STEP 4: PLOT 2 - OPTIMIZED FOR TAILS
# =============================================================================

def create_tails_plot(data, statistics, selected_station, output_dir):
    """
    Create distribution plot optimized to reveal TAILS
    Focuses on rare, extreme values (>100 μg/m³)
    """
    print("\n" + "="*80)
    print("STEP 4: CREATING PLOT 2 - OPTIMIZED FOR TAILS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== PLOT 2.1: Log-scale Histogram =====
    ax1 = axes[0, 0]
    
    # Create histogram with log y-scale to amplify rare events
    counts, bins, patches = ax1.hist(data, bins=50, alpha=0.7, 
                                      color='darkred', edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    
    ax1.axvline(EXTREME_THRESHOLD, color='black', linestyle='--', linewidth=2, 
                label=f'Extreme Threshold ({EXTREME_THRESHOLD})')
    ax1.axvline(statistics['percentiles'][99], color='blue', linestyle='--', linewidth=2, 
                label=f'99th percentile ({statistics["percentiles"][99]:.1f})')
    
    ax1.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
    ax1.set_title('Histogram with Log Y-Scale (Amplifies Rare Events)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    
    # ===== PLOT 2.2: CCDF (Complementary Cumulative Distribution) =====
    ax2 = axes[0, 1]
    
    # Sort data and calculate exceedance probabilities
    sorted_data = np.sort(data)
    ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    ax2.plot(sorted_data, ccdf, 'b-', linewidth=2, alpha=0.8)
    ax2.set_yscale('log')
    
    # Add threshold lines
    ax2.axvline(EXTREME_THRESHOLD, color='black', linestyle='--', linewidth=2, 
                label=f'Extreme Threshold')
    ax2.axhline(0.01, color='red', linestyle='--', linewidth=2, label='1% (99th percentile)')
    ax2.axvline(statistics['percentiles'][99], color='blue', linestyle='--', linewidth=2, 
                label=f'99th percentile')
    
    ax2.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('P(X > x) - Exceedance Probability (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('CCDF - Complementary Cumulative Distribution\n(Shows Tail Probabilities)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')
    
    # ===== PLOT 2.3: Q-Q Plot (Quantile-Quantile) =====
    ax3 = axes[1, 0]
    
    # Q-Q plot against normal distribution
    stats.probplot(data, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot Against Normal Distribution\n(Shows Deviation in Tails)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== PLOT 2.4: Zoomed Tail Region =====
    ax4 = axes[1, 1]
    
    # Focus only on values > 100
    tail_data = data[data > 100]
    if len(tail_data) > 0:
        ax4.hist(tail_data, bins=20, alpha=0.7, color='darkred', edgecolor='black')
        ax4.axvline(EXTREME_THRESHOLD, color='black', linestyle='--', linewidth=2)
        ax4.axvline(statistics['percentiles'][99], color='blue', linestyle='--', linewidth=2)
        
        ax4.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Zoomed Tail Region (PM2.5 > 100)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data in tail region (>100)', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    plt.suptitle(f'PLOT 2: DISTRIBUTIONS OPTIMIZED FOR TAILS\nStation {selected_station} - Focus on Extreme Values', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task3_tails_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Tails plot saved to: {output_path}")
    plt.show()
    plt.close()

# =============================================================================
# STEP 5: COMPARISON PLOT - PEAKS VS TAILS SIDE BY SIDE
# =============================================================================

def create_comparison_plot(data, statistics, selected_station, output_dir):
    """
    Create side-by-side comparison showing why tail optimization is needed
    """
    print("\n" + "="*80)
    print("STEP 5: CREATING COMPARISON PLOT (PEAKS VS TAILS)")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== LEFT: Standard histogram (what hides the tail) =====
    ax1 = axes[0]
    ax1.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(EXTREME_THRESHOLD, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('STANDARD HISTOGRAM\nTail is Invisible (Squashed)', fontsize=13, fontweight='bold', color='red')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation
    ax1.annotate('Tail region appears empty!', 
                xy=(EXTREME_THRESHOLD-50, 10), xytext=(EXTREME_THRESHOLD+50, 100),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # ===== RIGHT: Log-scale histogram (reveals tail) =====
    ax2 = axes[1]
    ax2.hist(data, bins=50, alpha=0.7, color='darkred', edgecolor='black')
    ax2.set_yscale('log')
    ax2.axvline(EXTREME_THRESHOLD, color='red', linestyle='--', linewidth=2)
    ax2.axvline(statistics['percentiles'][99], color='blue', linestyle='--', linewidth=2, 
                label=f'99th: {statistics["percentiles"][99]:.1f}')
    ax2.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('LOG-SCALE HISTOGRAM\nTail Revealed!', fontsize=13, fontweight='bold', color='green')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax2.annotate('Now we can see rare events!', 
                xy=(EXTREME_THRESHOLD-50, 2), xytext=(EXTREME_THRESHOLD+50, 10),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.suptitle('WHY TAIL-OPTIMIZED PLOTS ARE NECESSARY: Standard vs Log-Scale Histogram', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task3_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison plot saved to: {output_path}")
    plt.show()
    plt.close()

# =============================================================================
# STEP 6: TECHNICAL JUSTIFICATION
# =============================================================================

def generate_technical_justification(statistics, selected_station, output_dir):
    """
    Generate technical justification for which plot offers more honest depiction
    """
    print("\n" + "="*80)
    print("STEP 6: TECHNICAL JUSTIFICATION")
    print("="*80)
    
    p99 = statistics['percentiles'][99]
    p999 = statistics['percentiles'][99.9] if 99.9 in statistics['percentiles'] else p99
    
    justification = f"""
===============================================================================
TASK 3: TECHNICAL JUSTIFICATION - HONEST DEPICTION OF RARE EVENTS
===============================================================================

STATION ANALYZED: {selected_station}
EXTREME THRESHOLD: 200 μg/m³
99th PERCENTILE: {p99:.2f} μg/m³

1️⃣ PLOT 1: OPTIMIZED FOR PEAKS
------------------------------------------------------------
Visualization Methods Used:
• Histogram with optimal bin width (Freedman-Diaconis rule)
• Kernel Density Estimation (KDE)
• Box plot with outliers
• Violin plot

Strengths:
• Shows the main distribution shape clearly
• Reveals central tendency (mean, median, mode)
• Identifies if distribution is multimodal
• Appropriate bin width prevents under/over smoothing

Weaknesses:
• Tail region is compressed and hard to see
• Rare events (>200) appear as tiny bars or single outliers
• Cannot accurately assess probability of extreme events
• The "long tail" is visually minimized

2️⃣ PLOT 2: OPTIMIZED FOR TAILS
------------------------------------------------------------
Visualization Methods Used:
• Log-scale histogram (y-axis logarithmic)
• CCDF (Complementary Cumulative Distribution Function)
• Q-Q plot against normal distribution
• Zoomed tail region

Strengths:
• Log scale amplifies rare events, making them visible
• CCDF shows exact exceedance probabilities
• Reveals whether tail follows power law distribution
• Q-Q plot shows deviation from normality in tails
• Can accurately read probability of exceeding 200

Weaknesses:
• Main body of distribution appears compressed
• Not ideal for understanding typical values
• Requires understanding of log scales

🎯 WHICH PLOT OFFERS MORE "HONEST" DEPICTION?
------------------------------------------------------------
The **TAIL-OPTIMIZED PLOT (Plot 2)** offers a more honest depiction of
rare, hazardous events for the following reasons:

1. SCALE APPROPRIATENESS:
   • Standard plots make rare events appear insignificant
   • Log scale shows them in proportion to their probability
   • Example: Events at {p99:.1f} (99th percentile) occur 1% of the time
     - In Plot 1: Looks like almost zero
     - In Plot 2: Clearly visible as 10⁻² on y-axis

2. PROBABILITY VISUALIZATION:
   • CCDF directly shows P(X > x)
   • Can read exact probability of exceeding 200: {statistics['prob_extreme']:.6f}
   • This is impossible from standard histogram

3. TAIL BEHAVIOR ANALYSIS:
   • Q-Q plot shows if extreme events follow expected distribution
   • For this data, tail is {'heavier' if p999/p99 > 2 else 'lighter'} than normal
   • Important for risk assessment

4. BINNING ARTIFACTS:
   • Standard histograms hide rare events in last bin
   • Log-scale reveals internal structure of tail
   • No information loss about extreme values

EMPIRICAL EVIDENCE FROM THIS ANALYSIS:
------------------------------------------------------------
• Standard histogram: Tail appears empty beyond 150 μg/m³
• Log-scale histogram: Shows events up to {statistics['max']:.1f} μg/m³
• CCDF reveals that probability of exceeding 200 is exactly {statistics['prob_extreme']:.6f}
• This critical information is COMPLETELY MISSING from Plot 1

CONCLUSION:
------------------------------------------------------------
While Plot 1 is valuable for understanding typical pollution levels,
it is **fundamentally inadequate** for risk assessment of extreme events.
Plot 2 provides a mathematically honest representation by using
appropriate scales (log) and functions (CCDF) that preserve the
statistical significance of rare events.

The 99th percentile ({p99:.1f}) is {'ABOVE' if p99 > 200 else 'BELOW'} the extreme threshold,
meaning extreme events occur {'MORE' if p99 > 200 else 'LESS'} than 1% of the time.
This conclusion would be impossible to draw from Plot 1 alone.

===============================================================================
"""
    print(justification)
    
    # Save to file
    justification_path = os.path.join(output_dir, 'task3_technical_justification.txt')
    with open(justification_path, 'w', encoding='utf-8') as f:
        f.write(justification)
    
    print(f"✅ Technical justification saved to: {justification_path}")
    
    return justification

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function for Task 3"""
    try:
        print("\n" + "🚀"*40)
        print("🚀 TASK 3: DISTRIBUTION MODELING & TAIL INTEGRITY")
        print("🚀"*40 + "\n")
        
        # Load data
        df, script_dir = load_and_prepare_data()
        
        # Output directory
        output_dir = os.path.join(script_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Select industrial zone station
        selected_station, station_data, industrial_stations = select_industrial_zone(df)
        
        # Step 2: Calculate statistics - RENAMED to 'statistics'
        statistics = calculate_basic_stats(station_data, selected_station)
        
        # Step 3: Create peaks plot
        create_peaks_plot(station_data, statistics, selected_station, output_dir)
        
        # Step 4: Create tails plot
        create_tails_plot(station_data, statistics, selected_station, output_dir)
        
        # Step 5: Create comparison plot
        create_comparison_plot(station_data, statistics, selected_station, output_dir)
        
        # Step 6: Generate technical justification
        generate_technical_justification(statistics, selected_station, output_dir)
        
        print("\n" + "="*80)
        print("✅✅✅ TASK 3 COMPLETED SUCCESSFULLY ✅✅✅")
        print("="*80)
        print(f"\n📁 All outputs saved to: {output_dir}")
        print("\n📊 Generated files:")
        print(f"   1. {os.path.join(output_dir, 'task3_peaks_plot.png')} - Optimized for peaks")
        print(f"   2. {os.path.join(output_dir, 'task3_tails_plot.png')} - Optimized for tails")
        print(f"   3. {os.path.join(output_dir, 'task3_comparison.png')} - Side-by-side comparison")
        print(f"   4. {os.path.join(output_dir, 'task3_technical_justification.txt')} - Detailed justification")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()