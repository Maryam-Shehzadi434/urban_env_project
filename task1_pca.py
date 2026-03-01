"""
Task 1: Dimensionality Reduction Analysis
Urban Environmental Intelligence Challenge
Author: Data Architect
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def get_file_path():
    """
    Get the correct path to the data file regardless of where script is run from
    
    Returns:
    str: Absolute path to the parquet file
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build path relative to script location
    filepath = os.path.join(script_dir, 'data', 'raw', 'wide_format.parquet')
    
    return filepath, script_dir

def load_and_prepare_data():
    """
    Load the wide format data from raw directory and prepare for analysis
    
    Returns:
    pd.DataFrame: Prepared dataframe
    """
    print("="*80)
    print("TASK 1: DIMENSIONALITY REDUCTION CHALLENGE")
    print("="*80)
    
    # Get file path
    filepath, script_dir = get_file_path()
    
    print(f"\n📂 Script location: {script_dir}")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📂 Looking for data at: {filepath}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print("\n❌ ERROR: Data file not found!")
        print("\n🔍 Debug Information:")
        print(f"   Script directory: {script_dir}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Attempted path: {filepath}")
        
        print("\n📁 Expected directory structure:")
        print("   urban_env_project/")
        print("   ├── data/")
        print("   │   ├── raw/")
        print("   │   │   └── wide_format.parquet  <-- Your file")
        print("   │   └── processed/")
        print("   ├── task1_pca.py                 <-- This script")
        print("   └── ...")
        
        print("\n💡 Solutions:")
        print("   1. Make sure you're running the script from the correct location:")
        print(f"      cd {os.path.dirname(script_dir)}")
        print("      python urban_env_project/task1_pca.py")
        print("\n   2. Or run these commands:")
        print(f"      cd {script_dir}")
        print("      python task1_pca.py")
        
        sys.exit(1)
    
    # Load data
    print(f"\n✅ File found! Loading dataset...")
    df = pd.read_parquet(filepath)
    print(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    return df

def aggregate_by_station(df):
    """
    Aggregate hourly data to station-level averages
    
    Parameters:
    df (pd.DataFrame): Full dataset with hourly measurements
    
    Returns:
    pd.DataFrame: Station-level aggregated data
    """
    print("\n" + "="*80)
    print("STEP 1: AGGREGATING HOURLY DATA TO STATION LEVEL")
    print("="*80)
    print(f"\n📊 Original data: {len(df):,} hourly readings")
    print(f"📊 Number of unique stations: {df['station_id'].nunique()}")
    
    # Define columns to aggregate
    pollution_vars = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']
    metadata_vars = ['latitude', 'longitude', 'country', 'station_name']
    
    # Check which metadata columns exist
    available_metadata = [col for col in metadata_vars if col in df.columns]
    
    # Group by station_id and calculate means
    agg_dict = {var: 'mean' for var in pollution_vars}
    for var in available_metadata:
        agg_dict[var] = 'first'
    
    print(f"\n🔄 Aggregating {len(pollution_vars)} environmental variables...")
    station_df = df.groupby('station_id').agg(agg_dict).reset_index()
    
    print(f"\n✅ Aggregation complete:")
    print(f"   • {len(station_df)} stations")
    print(f"   • {len(station_df.columns)} columns")
    
    print("\n📈 Pollution variable ranges after aggregation:")
    for var in pollution_vars:
        print(f"   • {var}: {station_df[var].min():.2f} - {station_df[var].max():.2f} (mean: {station_df[var].mean():.2f})")
    
    return station_df

def create_zone_labels(station_df):
    """
    Create Industrial/Residential zone labels using KMeans clustering
    
    Parameters:
    station_df (pd.DataFrame): Station-level data
    
    Returns:
    pd.DataFrame: Data with zone labels added
    """
    print("\n" + "="*80)
    print("STEP 2: CLASSIFYING STATIONS INTO ZONES")
    print("="*80)
    
    # Features for classification
    pollution_features = ['pm25', 'pm10', 'no2']
    
    print(f"\n🏭 Using pollution features: {', '.join(pollution_features)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(station_df[pollution_features])
    
    # KMeans clustering with 2 clusters
    print("\n🔄 Applying K-Means clustering (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    station_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Determine which cluster is Industrial (higher pollution)
    cluster_means = station_df.groupby('cluster')[pollution_features].mean()
    industrial_cluster = cluster_means.mean(axis=1).idxmax()
    
    # Assign zone labels
    station_df['zone'] = station_df['cluster'].apply(
        lambda x: 'Industrial' if x == industrial_cluster else 'Residential'
    )
    
    # Print classification summary
    industrial_count = (station_df['zone'] == 'Industrial').sum()
    residential_count = (station_df['zone'] == 'Residential').sum()
    
    print(f"\n✅ Classification complete:")
    print(f"   • Industrial stations: {industrial_count} ({industrial_count/len(station_df)*100:.1f}%)")
    print(f"   • Residential stations: {residential_count} ({residential_count/len(station_df)*100:.1f}%)")
    
    print("\n📊 Zone pollution profiles (mean values):")
    zone_profile = station_df.groupby('zone')[pollution_features].mean()
    for zone in zone_profile.index:
        print(f"\n   {zone}:")
        for var in pollution_features:
            print(f"      {var}: {zone_profile.loc[zone, var]:.2f}")
    
    return station_df

def perform_pca_analysis(station_df):
    """
    Perform PCA on standardized environmental variables
    
    Parameters:
    station_df (pd.DataFrame): Station-level data with zone labels
    
    Returns:
    tuple: PCA results and transformed data
    """
    print("\n" + "="*80)
    print("STEP 3: PRINCIPAL COMPONENT ANALYSIS")
    print("="*80)
    
    # Select environmental variables
    env_vars = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'humidity']
    
    print(f"\n🔬 Analyzing {len(env_vars)} environmental variables:")
    print(f"   {', '.join(env_vars)}")
    
    # Standardize the data
    print("\n🔄 Standardizing variables (mean=0, std=1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(station_df[env_vars])
    
    # Apply PCA
    print("🔄 Applying PCA to reduce to 2 dimensions...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add PCA components to dataframe
    station_df['PC1'] = X_pca[:, 0]
    station_df['PC2'] = X_pca[:, 1]
    
    # Calculate variance explained
    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    
    print(f"\n✅ PCA completed:")
    print(f"   • PC1 explains {explained_var[0]:.1%} of variance")
    print(f"   • PC2 explains {explained_var[1]:.1%} of variance")
    print(f"   • Total variance explained: {cum_var[1]:.1%}")
    
    return pca, station_df, env_vars, explained_var

def analyze_pca_loadings(pca, env_vars):
    """
    Analyze and visualize PCA loadings
    
    Parameters:
    pca (PCA): Fitted PCA object
    env_vars (list): List of environmental variable names
    
    Returns:
    pd.DataFrame: Loadings dataframe
    """
    print("\n" + "="*80)
    print("STEP 4: ANALYZING PCA LOADINGS")
    print("="*80)
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=env_vars
    )
    
    print("\n🎯 Loadings (contribution of each variable to principal components):")
    print("\nPC1 Loadings (Primary pollution axis):")
    pc1_loadings = loadings['PC1'].sort_values(ascending=False)
    for var, val in pc1_loadings.items():
        direction = "positive" if val > 0 else "negative"
        magnitude = "strong" if abs(val) > 0.4 else "moderate" if abs(val) > 0.2 else "weak"
        print(f"   • {var}: {val:+.3f} ({direction}, {magnitude})")
    
    print("\nPC2 Loadings (Secondary axis):")
    pc2_loadings = loadings['PC2'].sort_values(ascending=False)
    for var, val in pc2_loadings.items():
        direction = "positive" if val > 0 else "negative"
        magnitude = "strong" if abs(val) > 0.4 else "moderate" if abs(val) > 0.2 else "weak"
        print(f"   • {var}: {val:+.3f} ({direction}, {magnitude})")
    
    # Identify dominant variables
    print("\n💡 Key Insights from Loadings:")
    dominant_pc1 = loadings['PC1'].abs().idxmax()
    dominant_pc2 = loadings['PC2'].abs().idxmax()
    print(f"   • PC1 is dominated by: {dominant_pc1} (loading = {loadings.loc[dominant_pc1, 'PC1']:+.3f})")
    print(f"   • PC2 is dominated by: {dominant_pc2} (loading = {loadings.loc[dominant_pc2, 'PC2']:+.3f})")
    
    return loadings

def create_pca_scatter_plot(station_df, explained_var, output_dir):
    """
    Create scatter plot of PCA results with zone coloring
    """
    plt.figure(figsize=(12, 10))
    
    colors = {'Industrial': '#FF6B6B', 'Residential': '#4ECDC4'}
    
    for zone in ['Industrial', 'Residential']:
        mask = station_df['zone'] == zone
        plt.scatter(station_df.loc[mask, 'PC1'], 
                   station_df.loc[mask, 'PC2'],
                   c=colors[zone], 
                   label=f'{zone} (n={mask.sum()})', 
                   alpha=0.7,
                   s=150,
                   edgecolors='black',
                   linewidth=1.5,
                   marker='o')
    
    # Add centroids
    for zone in ['Industrial', 'Residential']:
        mask = station_df['zone'] == zone
        centroid_x = station_df.loc[mask, 'PC1'].mean()
        centroid_y = station_df.loc[mask, 'PC2'].mean()
        plt.scatter(centroid_x, centroid_y, c='black', s=300, marker='X', 
                   edgecolors='white', linewidth=2, zorder=5)
        plt.annotate(f'{zone} Center', (centroid_x, centroid_y), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=14, fontweight='bold')
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=14, fontweight='bold')
    plt.title('Figure 1: PCA - Industrial vs Residential Zones', fontsize=16, fontweight='bold', pad=20)
    plt.legend(title='Zone Type', fontsize=12, title_fontsize=12, loc='best', framealpha=0.9)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f"Industrial Stations: {(station_df['zone']=='Industrial').sum()}\n"
    stats_text += f"Residential Stations: {(station_df['zone']=='Residential').sum()}\n"
    stats_text += f"Total Variance: {np.sum(explained_var):.1%}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'task1_pca_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PCA scatter plot saved to: {output_path}")
    plt.show()
    plt.close()

def create_loadings_heatmap(loadings, output_dir):
    """
    Create heatmap of PCA loadings
    """
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    ax = sns.heatmap(loadings.T, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.3f',
                cbar_kws={'label': 'Loading Value', 'shrink': 0.8},
                annot_kws={'size': 12},
                linewidths=1,
                linecolor='gray',
                square=True)
    
    plt.title('Figure 2: PCA Loadings Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Environmental Variables', fontsize=14, fontweight='bold')
    plt.ylabel('Principal Components', fontsize=14, fontweight='bold')
    
    # Add interpretation text
    plt.text(0.02, -0.15, 
             "Interpretation: Values close to ±1 indicate strong contribution to that PC",
             transform=plt.gca().transAxes, fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'task1_loadings_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Loadings heatmap saved to: {output_path}")
    plt.show()
    plt.close()

def create_variance_bar_plot(explained_var, output_dir):
    """
    Create bar plot of explained variance
    """
    plt.figure(figsize=(10, 8))
    
    components = [f'PC{i+1}' for i in range(len(explained_var))]
    colors_bars = ['#2E86AB', '#A23B72']
    bars = plt.bar(components, explained_var, color=colors_bars, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    plt.ylabel('Explained Variance Ratio', fontsize=14, fontweight='bold')
    plt.title('Figure 3: Variance Explained by Components', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(0, max(explained_var) * 1.2)
    
    # Add percentage labels on bars
    for bar, val in zip(bars, explained_var):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add cumulative variance text
    cum_var = np.cumsum(explained_var)[1]
    plt.text(0.5, 0.90, f'Cumulative: {cum_var:.1%}', 
             transform=plt.gca().transAxes, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'task1_variance_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Variance bar plot saved to: {output_path}")
    plt.show()
    plt.close()

def create_pc1_contributions_plot(loadings, output_dir):
    """
    Create horizontal bar chart of variable contributions to PC1
    """
    plt.figure(figsize=(12, 8))
    
    # Sort variables by absolute contribution to PC1
    contributions = loadings['PC1'].abs().sort_values(ascending=True)
    colors_contrib = plt.cm.Blues(np.linspace(0.4, 0.9, len(contributions)))
    
    y_pos = np.arange(len(contributions))
    bars = plt.barh(y_pos, contributions.values, color=colors_contrib, alpha=0.8, edgecolor='black', linewidth=1)
    plt.yticks(y_pos, contributions.index, fontsize=12)
    plt.xlabel('Absolute Loading Value (|PC1|)', fontsize=14, fontweight='bold')
    plt.title('Figure 4: Variable Contributions to PC1', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (v, bar) in enumerate(zip(contributions.values, bars)):
        plt.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'task1_pc1_contributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PC1 contributions plot saved to: {output_path}")
    plt.show()
    plt.close()

# =============================================================================
# NEW FUNCTION: PC2 Contributions Plot
# =============================================================================

def create_pc2_contributions_plot(loadings, output_dir):
    """
    Create horizontal bar chart of variable contributions to PC2
    This is the missing plot that was requested
    """
    plt.figure(figsize=(12, 8))
    
    # Sort variables by absolute contribution to PC2
    contributions = loadings['PC2'].abs().sort_values(ascending=True)
    colors_contrib = plt.cm.Oranges(np.linspace(0.4, 0.9, len(contributions)))
    
    y_pos = np.arange(len(contributions))
    bars = plt.barh(y_pos, contributions.values, color=colors_contrib, alpha=0.8, edgecolor='black', linewidth=1)
    plt.yticks(y_pos, contributions.index, fontsize=12)
    plt.xlabel('Absolute Loading Value (|PC2|)', fontsize=14, fontweight='bold')
    plt.title('Figure 5: Variable Contributions to PC2', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (v, bar) in enumerate(zip(contributions.values, bars)):
        plt.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # Add interpretation text
    plt.text(0.02, -0.15, 
             f"PC2 explains {loadings['PC2'].abs().sum()/len(loadings)*100:.1f}% of variance\n"
             "Dominant variable: " + loadings['PC2'].abs().idxmax(),
             transform=plt.gca().transAxes, fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'task1_pc2_contributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ PC2 contributions plot saved to: {output_path}")
    plt.show()
    plt.close()

def generate_analysis_summary(station_df, pca, loadings, explained_var, output_dir):
    """
    Generate a comprehensive analysis summary for the report
    
    Parameters:
    station_df (pd.DataFrame): Data with PCA results
    pca (PCA): Fitted PCA object
    loadings (pd.DataFrame): PCA loadings
    explained_var (array): Explained variance ratios
    output_dir (str): Directory to save summary
    """
    print("\n" + "="*80)
    print("📋 TASK 1 ANALYSIS SUMMARY")
    print("="*80)
    
    # Method justification
    print("\n🎯 DIMENSIONALITY REDUCTION METHOD: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 60)
    print("Justification for choosing PCA:")
    print("   1. Linear relationships: Environmental variables often have linear correlations")
    print("   2. Interpretability: PCA provides clear loadings for axis interpretation")
    print("   3. Variance preservation: Maximizes retained variance in first two components")
    print("   4. Standardization compatible: Works well with standardized data (required here)")
    print("   5. No distributional assumptions: Makes no assumptions about data distribution")
    print("   6. Widely accepted: Industry standard for environmental data analysis")
    
    print("\n   Alternative methods considered and rejected:")
    print("   • t-SNE: Non-linear, loses global structure, loadings not interpretable")
    print("   • UMAP: Similar to t-SNE, loadings cannot be analyzed")
    print("   • Factor Analysis: More assumptions, less variance-focused")
    print("   • MDS: Distance-based, less interpretable for variable contributions")
    
    # Loadings interpretation
    print("\n📊 PRINCIPAL COMPONENT LOADINGS INTERPRETATION")
    print("-" * 60)
    print(f"PC1 (explains {explained_var[0]:.1%} of variance):")
    print("   → Represents the 'Overall Pollution Intensity' axis")
    pc1_top = loadings['PC1'].abs().nlargest(3)
    for var in pc1_top.index:
        sign = "+" if loadings.loc[var, 'PC1'] > 0 else "-"
        impact = "increases" if loadings.loc[var, 'PC1'] > 0 else "decreases"
        print(f"   • {var}: {sign}{abs(loadings.loc[var, 'PC1']):.3f} ({impact} PC1)")
    
    print(f"\nPC2 (explains {explained_var[1]:.1%} of variance):")
    print("   → Represents the 'Pollution Composition & Weather' axis")
    pc2_top = loadings['PC2'].abs().nlargest(3)
    for var in pc2_top.index:
        sign = "+" if loadings.loc[var, 'PC2'] > 0 else "-"
        impact = "increases" if loadings.loc[var, 'PC2'] > 0 else "decreases"
        print(f"   • {var}: {sign}{abs(loadings.loc[var, 'PC2']):.3f} ({impact} PC2)")
    
    # Key drivers of urban pollution
    print("\n🏭 MAIN DRIVERS OF URBAN POLLUTION")
    print("-" * 60)
    print("Based on PCA loadings analysis and zone comparison:")
    
    # Calculate statistics for each zone
    pollution_vars = ['pm25', 'pm10', 'no2']
    industrial_stats = station_df[station_df['zone'] == 'Industrial'][pollution_vars].mean()
    residential_stats = station_df[station_df['zone'] == 'Residential'][pollution_vars].mean()
    
    print("\n   Pollution level comparison (Industrial vs Residential):")
    for var in pollution_vars:
        diff = industrial_stats[var] - residential_stats[var]
        pct_diff = (diff / residential_stats[var] * 100)
        print(f"   • {var.upper()}:")
        print(f"     - Industrial: {industrial_stats[var]:.1f}")
        print(f"     - Residential: {residential_stats[var]:.1f}")
        print(f"     - Difference: {diff:+.1f} ({pct_diff:+.0f}%)")
    
    # Determine primary drivers from loadings
    primary_driver = loadings['PC1'].abs().idxmax()
    secondary_driver = loadings['PC2'].abs().idxmax()
    
    print(f"\n🔍 Key Findings:")
    print(f"   • Primary pollution driver: {primary_driver.upper()} (dominates PC1)")
    print(f"   • Secondary distinguishing factor: {secondary_driver.upper()} (dominates PC2)")
    print(f"   • Industrial zones cluster at {station_df[station_df['zone']=='Industrial']['PC1'].mean():+.2f} on PC1")
    print(f"   • Residential zones cluster at {station_df[station_df['zone']=='Residential']['PC1'].mean():+.2f} on PC1")
    
    # Zone separation analysis
    print("\n📍 INDUSTRIAL VS RESIDENTIAL SEPARATION")
    print("-" * 60)
    
    industrial_pc1 = station_df[station_df['zone'] == 'Industrial']['PC1']
    residential_pc1 = station_df[station_df['zone'] == 'Residential']['PC1']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(industrial_pc1, residential_pc1)
    
    print(f"   Statistical test of separation (PC1):")
    print(f"   • t-statistic: {t_stat:.3f}")
    print(f"   • p-value: {p_value:.3e}")
    print(f"   • Conclusion: {'Significant' if p_value < 0.05 else 'Not significant'} separation between zones")
    
    print("\n💡 ACTIONABLE INSIGHTS FOR URBAN PLANNING:")
    print("   • Target PM2.5 and PM10 reduction in industrial zones first")
    print("   • Monitor NO2 as key tracer of industrial activity")
    print("   • O3 shows different pattern (often higher in residential due to NO titration)")
    print("   • Use PC1 as composite index for pollution monitoring")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'task1_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TASK 1: DIMENSIONALITY REDUCTION ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Stations: {len(station_df)}\n")
        f.write(f"Industrial Stations: {(station_df['zone']=='Industrial').sum()}\n")
        f.write(f"Residential Stations: {(station_df['zone']=='Residential').sum()}\n\n")
        f.write(f"PC1 Variance: {explained_var[0]:.1%}\n")
        f.write(f"PC2 Variance: {explained_var[1]:.1%}\n")
        f.write(f"Total Variance: {np.cumsum(explained_var)[1]:.1%}\n\n")
        f.write("Loadings Matrix:\n")
        f.write(loadings.to_string())
    
    print(f"\n✅ Summary saved to: {summary_path}")

def save_results(station_df, loadings, explained_var, output_dir):
    """
    Save analysis results to CSV files
    
    Parameters:
    station_df (pd.DataFrame): Data with PCA results
    loadings (pd.DataFrame): PCA loadings
    explained_var (array): Explained variance ratios
    output_dir (str): Directory to save results
    """
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    # Save station-level data with PCA results
    station_output = os.path.join(output_dir, 'task1_station_data.csv')
    station_df.to_csv(station_output, index=False)
    print(f"✅ Station data saved to: {station_output}")
    
    # Save loadings
    loadings_output = os.path.join(output_dir, 'task1_pca_loadings.csv')
    loadings.to_csv(loadings_output)
    print(f"✅ PCA loadings saved to: {loadings_output}")
    
    # Save summary statistics
    summary = pd.DataFrame({
        'Metric': [
            'Number of Stations',
            'Industrial Stations',
            'Residential Stations',
            'Industrial %',
            'Residential %',
            'PC1 Variance %',
            'PC2 Variance %',
            'Total Variance %',
            'Primary Driver (PC1)',
            'Primary Driver Loading',
            'Secondary Driver (PC2)',
            'Secondary Driver Loading'
        ],
        'Value': [
            len(station_df),
            (station_df['zone'] == 'Industrial').sum(),
            (station_df['zone'] == 'Residential').sum(),
            f"{(station_df['zone'] == 'Industrial').sum()/len(station_df)*100:.1f}%",
            f"{(station_df['zone'] == 'Residential').sum()/len(station_df)*100:.1f}%",
            f"{explained_var[0]:.1%}",
            f"{explained_var[1]:.1%}",
            f"{np.cumsum(explained_var)[1]:.1%}",
            loadings['PC1'].abs().idxmax(),
            f"{loadings.loc[loadings['PC1'].abs().idxmax(), 'PC1']:.3f}",
            loadings['PC2'].abs().idxmax(),
            f"{loadings.loc[loadings['PC2'].abs().idxmax(), 'PC2']:.3f}"
        ]
    })
    
    summary_output = os.path.join(output_dir, 'task1_summary.csv')
    summary.to_csv(summary_output, index=False)
    print(f"✅ Summary statistics saved to: {summary_output}")

def main():
    """
    Main execution function for Task 1
    """
    try:
        print("\n" + "🚀"*40)
        print("🚀 STARTING TASK 1: DIMENSIONALITY REDUCTION ANALYSIS")
        print("🚀"*40 + "\n")
        
        # Execute pipeline
        df = load_and_prepare_data()
        station_df = aggregate_by_station(df)
        station_df = create_zone_labels(station_df)
        pca, station_df, env_vars, explained_var = perform_pca_analysis(station_df)
        loadings = analyze_pca_loadings(pca, env_vars)
        
        # Get output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create individual visualizations
        print("\n" + "="*80)
        print("STEP 5: CREATING INDIVIDUAL VISUALIZATIONS")
        print("="*80)
        
        create_pca_scatter_plot(station_df, explained_var, output_dir)
        create_loadings_heatmap(loadings, output_dir)
        create_variance_bar_plot(explained_var, output_dir)
        create_pc1_contributions_plot(loadings, output_dir)
        create_pc2_contributions_plot(loadings, output_dir)  # NEW: PC2 contributions plot
        
        # Generate analysis summary and save results
        generate_analysis_summary(station_df, pca, loadings, explained_var, output_dir)
        save_results(station_df, loadings, explained_var, output_dir)
        
        print("\n" + "="*80)
        print("✅✅✅ TASK 1 COMPLETED SUCCESSFULLY ✅✅✅")
        print("="*80)
        print(f"\n📁 All outputs saved to: {output_dir}")
        print("\n📊 Generated files:")
        print(f"   • {os.path.join(output_dir, 'task1_pca_scatter.png')} - PCA scatter plot")
        print(f"   • {os.path.join(output_dir, 'task1_loadings_heatmap.png')} - Loadings heatmap")
        print(f"   • {os.path.join(output_dir, 'task1_variance_bar.png')} - Variance bar plot")
        print(f"   • {os.path.join(output_dir, 'task1_pc1_contributions.png')} - PC1 contributions")
        print(f"   • {os.path.join(output_dir, 'task1_pc2_contributions.png')} - PC2 contributions")  # NEW
        print(f"   • {os.path.join(output_dir, 'task1_station_data.csv')} - Station data with PCA results")
        print(f"   • {os.path.join(output_dir, 'task1_pca_loadings.csv')} - PCA loadings matrix")
        print(f"   • {os.path.join(output_dir, 'task1_summary.csv')} - Summary statistics")
        print(f"   • {os.path.join(output_dir, 'task1_analysis_summary.txt')} - Detailed analysis summary")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()