# feature_engineering.py (OPTIMIZED - Won't get stuck)
# ==============================
# PURPOSE: Convert long to wide format efficiently
# ==============================

import pandas as pd
import numpy as np
import os

INPUT_FILE = r"D:\DataScience Assignments\urban_env_project\data\raw\Long_format.csv"
OUTPUT_FILE = r"D:\DataScience Assignments\urban_env_project\data\raw\wide_format.parquet"

print("=" * 80)
print("FEATURE ENGINEERING - Long to Wide Conversion")
print("=" * 80)

# ==============================
# STEP 1: LOAD DATA
# ==============================
print("\n📂 STEP 1: Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df):,} rows")
print(f"✅ Columns: {list(df.columns)}")

# ==============================
# STEP 2: CLEAN AND PREPARE
# ==============================
print("\n🔄 STEP 2: Cleaning and preparing data...")

# Clean parameter names
df['parameter'] = df['parameter'].str.strip().str.lower()

# Convert types
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['station_id'] = pd.to_numeric(df['station_id'], errors='coerce').astype('Int64')

print(f"✅ Data prepared")

# ==============================
# STEP 3: SPLIT DATA BY TYPE (FAST!)
# ==============================
print("\n🔄 STEP 3: Splitting data by type...")

# Weather data (has timestamps)
weather_data = df[df['parameter'].isin(['temperature', 'humidity'])].copy()
print(f"✅ Weather data: {len(weather_data):,} rows with timestamps")

# Pollution data (missing timestamps)
pollution_data = df[df['parameter'].isin(['no2', 'o3', 'pm10', 'pm25'])].copy()
print(f"✅ Pollution data: {len(pollution_data):,} rows (will be matched by position)")

# ==============================
# STEP 4: GET WEATHER TIMESTAMPS BY STATION
# ==============================
print("\n🔄 STEP 4: Getting weather timestamps by station...")

# Get unique weather timestamps per station
weather_timestamps = {}
for station in weather_data['station_id'].unique():
    station_weather = weather_data[weather_data['station_id'] == station]
    timestamps = station_weather['datetime'].dropna().unique()
    # Sort timestamps to maintain order
    timestamps = np.sort(timestamps)
    weather_timestamps[station] = timestamps
    print(f"   Station {station}: {len(timestamps)} weather timestamps")

# ==============================
# STEP 5: ASSIGN TIMESTAMPS TO POLLUTION DATA (VECTORIZED - FAST!)
# ==============================
print("\n🔄 STEP 5: Assigning timestamps to pollution data...")

# Create a list to hold processed pollution data
processed_pollution = []

for station in pollution_data['station_id'].unique():
    # Get pollution data for this station
    station_pollution = pollution_data[pollution_data['station_id'] == station].copy()
    
    if station in weather_timestamps and len(weather_timestamps[station]) > 0:
        # Get weather timestamps for this station
        station_times = weather_timestamps[station]
        
        # Repeat timestamps to match pollution data length
        n_pollution = len(station_pollution)
        n_times = len(station_times)
        
        # Calculate how many times to repeat
        repeats = (n_pollution + n_times - 1) // n_times
        
        # Create repeated timestamps
        repeated_times = np.tile(station_times, repeats)[:n_pollution]
        
        # Assign to pollution data
        station_pollution['datetime'] = repeated_times
        
        processed_pollution.append(station_pollution)
        print(f"   Station {station}: assigned {n_pollution} pollution readings to {n_times} weather timestamps")
    else:
        print(f"   ⚠️ Station {station}: no weather data found, keeping original timestamps")

# Combine processed pollution data
if processed_pollution:
    pollution_data_filled = pd.concat(processed_pollution, ignore_index=True)
else:
    pollution_data_filled = pollution_data

print(f"✅ Assigned timestamps to {len(pollution_data_filled):,} pollution rows")

# ==============================
# STEP 6: COMBINE ALL DATA
# ==============================
print("\n🔄 STEP 6: Combining all data...")

# Combine weather and filled pollution data
df_filled = pd.concat([weather_data, pollution_data_filled], ignore_index=True)
print(f"✅ Combined data: {len(df_filled):,} rows")

# ==============================
# STEP 7: CREATE WIDE FORMAT
# ==============================
print("\n🔄 STEP 7: Creating wide format...")

# Pivot to wide format
df_wide = df_filled.pivot_table(
    index=['station_id', 'datetime', 'station_name', 'country', 'latitude', 'longitude'],
    columns='parameter',
    values='value',
    aggfunc='first'
).reset_index()

# Clean up column names
df_wide.columns.name = None

print(f"✅ Wide format: {len(df_wide):,} rows × {len(df_wide.columns)} columns")

# ==============================
# STEP 8: CHECK STATION 162 SPECIFICALLY
# ==============================
print("\n🔍 STEP 8: Checking station 162 (Houston Deer)...")

station_162 = df_wide[df_wide['station_id'] == 162].copy()
print(f"   Station 162 has {len(station_162)} rows in wide format")

if len(station_162) > 0:
    print("\n   Sample of station 162 data (first 10 rows):")
    display_cols = ['datetime', 'temperature', 'humidity', 'no2', 'o3', 'pm10', 'pm25']
    print(station_162[display_cols].head(10).to_string())
    
    # Show non-null counts for station 162
    print("\n   Station 162 data completeness:")
    for param in ['no2', 'o3', 'pm10', 'pm25', 'temperature', 'humidity']:
        if param in station_162.columns:
            non_null = station_162[param].notna().sum()
            print(f"      {param}: {non_null}/{len(station_162)} rows ({non_null/len(station_162)*100:.1f}%)")

# ==============================
# STEP 9: CREATE TIME FEATURES
# ==============================
print("\n⏰ STEP 9: Creating time features...")

df_wide['year'] = df_wide['datetime'].dt.year
df_wide['month'] = df_wide['datetime'].dt.month
df_wide['day'] = df_wide['datetime'].dt.day
df_wide['hour'] = df_wide['datetime'].dt.hour
df_wide['day_of_week'] = df_wide['datetime'].dt.dayofweek
df_wide['is_weekend'] = (df_wide['day_of_week'] >= 5).astype(int)

# ==============================
# STEP 10: SORT AND SAVE
# ==============================
print("\n📊 STEP 10: Sorting and saving...")
df_wide = df_wide.sort_values(['station_id', 'datetime']).reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_wide.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
print(f"✅ Saved to: {OUTPUT_FILE}")

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 80)

print(f"\n📊 Final dataset: {df_wide.shape[0]:,} rows × {df_wide.shape[1]} columns")
print(f"📍 Unique stations: {df_wide['station_id'].nunique()}")

# Overall parameter coverage
print(f"\n📈 Overall Parameter Coverage:")
for param in ['temperature', 'humidity', 'no2', 'o3', 'pm10', 'pm25']:
    if param in df_wide.columns:
        non_null = df_wide[param].notna().sum()
        pct = (non_null / len(df_wide)) * 100
        print(f"  • {param:12}: {non_null:>8,} / {len(df_wide):,} rows ({pct:>5.1f}%)")

print("\n🎯 Ready for next tasks!")