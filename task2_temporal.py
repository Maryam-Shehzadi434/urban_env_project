"""
Task 2: High-Density Temporal Analysis - MINIMAL VERSION
3 Essential Visualizations Only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuration
THRESHOLD = 35
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_file_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'data', 'raw', 'wide_format.parquet'), script_dir

def load_and_prepare_data():
    print("="*80)
    print("TASK 2: HIGH-DENSITY TEMPORAL ANALYSIS")
    print("="*80)
    
    filepath, script_dir = get_file_path()
    print(f"\n📂 Loading data from: {filepath}")
    
    df = pd.read_parquet(filepath)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Create violation flag
    df['violation'] = (df['pm25'] > THRESHOLD).astype(int)
    
    # Extract temporal features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.date
    
    print(f"✅ Created violation flag: {df['violation'].sum():,} violations ({df['violation'].mean()*100:.1f}%)")
    
    return df, script_dir

# =============================================================================
# VISUALIZATION 1: HEATMAP (Time × Station) - MAIN HIGH-DENSITY PLOT
# =============================================================================

def create_heatmap(df, output_dir):
    """
    Heatmap showing all 100 stations over time
    X-axis: Time (days)
    Y-axis: Station ID
    Color: Violation rate (0-100%)
    """
    print("\n" + "="*80)
    print("VISUALIZATION 1: HIGH-DENSITY HEATMAP")
    print("="*80)
    
    # Aggregate to daily violation rate per station
    daily_violations = df.groupby(['station_id', 'date'])['violation'].mean() * 100
    
    # Pivot for heatmap
    heatmap_data = daily_violations.reset_index().pivot(
        index='station_id', 
        columns='date', 
        values='violation'
    )
    
    # Sort stations by total violation rate (worst at top)
    station_order = daily_violations.groupby('station_id').mean().sort_values(ascending=False).index
    heatmap_data = heatmap_data.loc[station_order]
    
    print(f"📊 Heatmap shape: {heatmap_data.shape[0]} stations × {heatmap_data.shape[1]} days")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Custom colormap (blue = safe, yellow = moderate, red = high violation)
    colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
              '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap=cmap, 
                   interpolation='nearest', vmin=0, vmax=100)
    
    # Labels and title
    ax.set_xlabel('Day of Year (Jan → Dec)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Station ID (sorted by violation rate)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: PM2.5 Violation Heatmap - All 100 Stations Over Time\n(Darker red = higher violation rate)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis: show month labels
    month_days = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Find indices in our data
    day_indices = []
    for i, days in enumerate(month_days):
        if days < len(heatmap_data.columns):
            day_indices.append(days)
    
    ax.set_xticks(day_indices)
    ax.set_xticklabels(month_names[:len(day_indices)], rotation=45, ha='right')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Daily Violation Rate (%)', fontsize=11, fontweight='bold')
    
    # Add horizontal line separating top 10 violators
    top_10_idx = int(len(heatmap_data) * 0.1)
    ax.axhline(y=top_10_idx, color='white', linewidth=2, linestyle='-', alpha=0.8)
    ax.text(len(heatmap_data.columns) - 10, top_10_idx + 2, 
            'Top 10% Violators', color='white', fontweight='bold', ha='right')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task2_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.show()
    plt.close()
    
    return heatmap_data

# =============================================================================
# VISUALIZATION 2: DAILY PATTERN (24-Hour Cycle)
# =============================================================================

def create_daily_pattern(df, output_dir):
    """
    Line plot showing average violation rate by hour of day
    Shows daily periodic signature
    """
    print("\n" + "="*80)
    print("VISUALIZATION 2: DAILY PATTERN (24-HOUR CYCLE)")
    print("="*80)
    
    # Calculate violation rate by hour
    hourly = df.groupby('hour')['violation'].mean() * 100
    
    print("\n📊 HOURLY VIOLATION RATES:")
    for hour in range(0, 24, 3):
        print(f"   Hour {hour:02d}:00 → {hourly[hour]:.1f}%")
    print(f"   Morning peak (8 AM): {hourly[8]:.1f}%")
    print(f"   Evening peak (6 PM): {hourly[18]:.1f}%")
    print(f"   Overnight low (3 AM): {hourly[3]:.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line with markers
    ax.plot(hourly.index, hourly.values, marker='o', linewidth=2.5, 
            markersize=8, color='#d73027', markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor='#d73027')
    
    # Fill area under curve
    ax.fill_between(hourly.index, 0, hourly.values, alpha=0.3, color='#d73027')
    
    # Highlight rush hours
    ax.axvspan(6, 9, alpha=0.2, color='blue', label='Morning Rush (6-9 AM)')
    ax.axvspan(16, 19, alpha=0.2, color='orange', label='Evening Rush (4-7 PM)')
    
    # Labels and title
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Daily Pattern - PM2.5 Violations by Hour of Day\n(24-hour cycle)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis: show all hours
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    
    # Y-axis: set reasonable limits
    ax.set_ylim(0, max(hourly) * 1.2)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    
    # Add annotation for peak values
    ax.annotate(f'Morning Peak: {hourly[8]:.1f}%', 
                xy=(8, hourly[8]), xytext=(10, hourly[8]+2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    
    ax.annotate(f'Evening Peak: {hourly[18]:.1f}%', 
                xy=(18, hourly[18]), xytext=(20, hourly[18]+2),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task2_daily_pattern.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Daily pattern plot saved to: {output_path}")
    plt.show()
    plt.close()
    
    return hourly

# =============================================================================
# VISUALIZATION 3: MONTHLY PATTERN (Seasonal Cycle)
# =============================================================================

def create_monthly_pattern(df, output_dir):
    """
    Bar/line plot showing average violation rate by month
    Shows monthly periodic signature
    """
    print("\n" + "="*80)
    print("VISUALIZATION 3: MONTHLY PATTERN (SEASONAL CYCLE)")
    print("="*80)
    
    # Calculate violation rate by month
    monthly = df.groupby('month')['violation'].mean() * 100
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly.index = month_names
    
    print("\n📊 MONTHLY VIOLATION RATES:")
    for month in month_names:
        print(f"   {month}: {monthly[month]:.1f}%")
    print(f"   Winter average (Dec-Feb): {monthly[['Dec','Jan','Feb']].mean():.1f}%")
    print(f"   Summer average (Jun-Aug): {monthly[['Jun','Jul','Aug']].mean():.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars with color gradient based on value
    colors = []
    for month in month_names:
        if monthly[month] < 10:
            colors.append('#313695')  # dark blue
        elif monthly[month] < 15:
            colors.append('#4575b4')  # blue
        elif monthly[month] < 20:
            colors.append('#74add1')  # light blue
        elif monthly[month] < 25:
            colors.append('#fdae61')  # orange
        else:
            colors.append('#d73027')  # red
    
    bars = ax.bar(month_names, monthly.values, color=colors, 
                  edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add a line to show trend
    ax.plot(month_names, monthly.values, color='black', 
            linewidth=2, marker='o', markersize=6, linestyle='--', alpha=0.5)
    
    # Highlight winter and summer
    ax.axvspan('Dec', 'Feb', alpha=0.1, color='blue', label='Winter')
    ax.axvspan('Jun', 'Aug', alpha=0.1, color='orange', label='Summer')
    
    # Labels and title
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Monthly Pattern - PM2.5 Violations by Month\n(Seasonal cycle)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar, val in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, max(monthly) * 1.2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'task2_monthly_pattern.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Monthly pattern plot saved to: {output_path}")
    plt.show()
    plt.close()
    
    return monthly

# =============================================================================
# SUMMARY AND CONCLUSION
# =============================================================================

def generate_summary(df, hourly, monthly, output_dir):
    """
    Generate concise summary with conclusion
    """
    print("\n" + "="*80)
    print("TASK 2: ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate key metrics
    total_violations = df['violation'].sum()
    violation_rate = df['violation'].mean() * 100
    
    daily_peak_hour = hourly.idxmax()
    daily_peak_rate = hourly.max()
    
    winter_avg = monthly[['Dec', 'Jan', 'Feb']].mean()
    summer_avg = monthly[['Jun', 'Jul', 'Aug']].mean()
    seasonal_diff = winter_avg - summer_avg
    
    # Determine primary driver
    if daily_peak_rate > 25 and abs(seasonal_diff) > 10:
        driver = "BOTH daily traffic cycles AND seasonal patterns"
        evidence = f"Daily: {daily_peak_rate:.1f}% during rush hour\nSeasonal: {abs(seasonal_diff):.1f}% difference between winter and summer"
    elif daily_peak_rate > 25:
        driver = "DAILY traffic cycles (24-hour pattern)"
        evidence = f"Clear rush hour peaks: {daily_peak_rate:.1f}% at {daily_peak_hour}:00"
    elif abs(seasonal_diff) > 10:
        driver = "MONTHLY seasonal patterns (winter heating/inversions)"
        evidence = f"Winter {winter_avg:.1f}% vs Summer {summer_avg:.1f}%"
    else:
        driver = "No strong periodic pattern detected"
        evidence = "Relatively flat daily and monthly patterns"
    
    summary = f"""
===============================================================================
TASK 2: HIGH-DENSITY TEMPORAL ANALYSIS - CONCLUSION
===============================================================================

OVERALL STATISTICS:
• Total violations: {total_violations:,} hours
• Overall violation rate: {violation_rate:.1f}%
• Stations analyzed: {df['station_id'].nunique()}

DAILY PATTERN (24-HOUR CYCLE):
• Peak hour: {daily_peak_hour}:00 ({daily_peak_rate:.1f}%)
• Morning rush (7-9 AM) average: {hourly.loc[7:9].mean():.1f}%
• Evening rush (5-7 PM) average: {hourly.loc[17:19].mean():.1f}%
• Overnight (12-4 AM) average: {hourly.loc[0:4].mean():.1f}%

MONTHLY PATTERN (SEASONAL CYCLE):
• Winter average (Dec-Feb): {winter_avg:.1f}%
• Summer average (Jun-Aug): {summer_avg:.1f}%
• Seasonal difference: {seasonal_diff:+.1f}% (winter vs summer)

PRIMARY DRIVER: {driver}

EVIDENCE:
{evidence}

===============================================================================
"""
    
    print(summary)
    
    # Save summary - use utf-8 encoding to handle any special characters
    summary_path = os.path.join(output_dir, 'task2_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✅ Summary saved to: {summary_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    try:
        print("\n" + "*"*40)
        print("TASK 2: MINIMAL VIABLE VISUALIZATIONS (3 PLOTS)")
        print("*"*40 + "\n")
        
        # Load data
        df, script_dir = load_and_prepare_data()
        
        # Output directory
        output_dir = os.path.join(script_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create 3 essential visualizations
        heatmap_data = create_heatmap(df, output_dir)
        hourly_pattern = create_daily_pattern(df, output_dir)
        monthly_pattern = create_monthly_pattern(df, output_dir)
        
        # Generate summary
        generate_summary(df, hourly_pattern, monthly_pattern, output_dir)
        
        print("\n" + "="*80)
        print("TASK 2 COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\n📁 3 ESSENTIAL VISUALIZATIONS CREATED:")
        print(f"   1. Heatmap (all 100 stations over time)")
        print(f"   2. Daily pattern (24-hour cycle)")
        print(f"   3. Monthly pattern (seasonal cycle)")
        print(f"\n   Location: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()