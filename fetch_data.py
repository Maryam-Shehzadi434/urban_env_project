import requests
import time
import pandas as pd
import os
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================

API_KEY = "ca493f141a53c46a989734d01c477bea314f68d70a7bb2c7bf8482b839b7041a"
HEADERS = {"X-API-Key": API_KEY}
BASE_URL = "https://api.openaq.org/v3"

# Full year 2025
DATE_FROM = "2025-01-01T00:00:00Z"
DATE_TO = "2025-12-31T23:59:59Z"

REQUIRED_POLLUTION = ["pm25", "pm10", "no2", "o3"]
PREFERRED_COUNTRIES = ["US", "CA", "DE", "FR", "GB", "NL", "ES", "IT", "JP", "AU"]
TARGET_STATIONS = 100

# File paths
OUTPUT_FILE = "data/progress.csv"
os.makedirs("data", exist_ok=True)

# ==============================
# LOAD EXISTING STATIONS (MEMORY EFFICIENT)
# ==============================

def load_existing_stations():
    """Load station IDs WITHOUT loading full file into memory"""
    
    if not os.path.exists(OUTPUT_FILE):
        print("📂 No existing file found - starting fresh")
        return set(), 0
    
    try:
        # Read ONLY the station_id column - memory efficient
        df = pd.read_csv(OUTPUT_FILE, usecols=['station_id'])
        existing_ids = set(df['station_id'].unique())
        station_count = len(existing_ids)
        
        # Get row count efficiently (without loading data)
        with open(OUTPUT_FILE, 'r') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header
        
        print("=" * 70)
        print("📊 EXISTING DATA SUMMARY")
        print("=" * 70)
        print(f"📁 File: {OUTPUT_FILE}")
        print(f"📊 Total rows: {row_count:,}")
        print(f"📍 Stations found: {station_count}")
        print(f"🎯 Next station will be: #{station_count + 1}")
        print("=" * 70)
        
        return existing_ids, station_count
        
    except Exception as e:
        print(f"⚠️ Error reading file: {e}")
        return set(), 0

# ==============================
# YOUR EXISTING FETCH FUNCTIONS (UNCHANGED)
# ==============================

def fetch_all_measurements(sensor_id, param_name):
    """Fetch all measurements for a sensor in 2025"""
    
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
    all_records = []
    page = 1
    
    while True:
        params = {
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "limit": 1000,
            "page": page
        }
        
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            
            if response.status_code != 200:
                break
                
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                break
                
            for record in results:
                all_records.append({
                    "datetime": record.get("datetime", {}).get("utc"),
                    "parameter": param_name,
                    "value": record.get("value"),
                    "unit": record.get("unit")
                })
            
            if len(results) < 1000:
                break
                
            page += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    return all_records

def fetch_weather_data(lat, lon):
    """Fetch temperature and humidity for 2025"""
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
            
        data = response.json()
        hourly = data.get("hourly", {})
        
        weather_records = []
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humids = hourly.get("relative_humidity_2m", [])
        
        for i in range(len(times)):
            weather_records.append({
                "datetime": times[i],
                "parameter": "temperature",
                "value": temps[i] if i < len(temps) else None,
                "unit": "°C"
            })
            weather_records.append({
                "datetime": times[i],
                "parameter": "humidity",
                "value": humids[i] if i < len(humids) else None,
                "unit": "%"
            })
        
        return weather_records
        
    except Exception as e:
        print(f"    Weather error: {e}")
        return []

def get_country_id(code):
    """Map country codes to OpenAQ country IDs"""
    country_map = {
        "US": 155, "CA": 40, "DE": 81, "FR": 75,
        "GB": 225, "NL": 156, "ES": 70, "IT": 109,
        "JP": 112, "AU": 14
    }
    return country_map.get(code, 155)

# ==============================
# PROFESSIONAL APPEND-BASED COLLECTION
# ==============================

def collect_remaining_stations():
    """Collect remaining stations and APPEND efficiently"""
    
    # Load existing stations (memory efficient)
    existing_ids, completed_count = load_existing_stations()
    
    if completed_count >= TARGET_STATIONS:
        print("\n🎉 All 100 stations already collected!")
        return
    
    remaining = TARGET_STATIONS - completed_count
    print(f"\n🎯 Need to collect {remaining} more stations (from #{completed_count + 1} to #100)")
    
    # Buffer for current batch of stations (max 10)
    current_batch = []
    stations_in_batch = 0
    total_new_stations = 0
    
    # Check if file exists for header decision
    file_exists = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
    
    for country in PREFERRED_COUNTRIES:
        if (completed_count + total_new_stations) >= TARGET_STATIONS:
            break
            
        print(f"\n🌍 Scanning {country}...")
        
        # Get locations for this country
        params = {
            "countries_id": get_country_id(country),
            "parameters_id": "2,1,7,10",
            "limit": 50,
            "sort": "desc"
        }
        
        response = requests.get(f"{BASE_URL}/locations", headers=HEADERS, params=params)
        
        if response.status_code != 200:
            continue
            
        locations = response.json().get("results", [])
        
        for location in locations:
            if (completed_count + total_new_stations) >= TARGET_STATIONS:
                break
            
            location_id = location.get('id')
            location_name = location.get('name')
            
            # ✅ ROBUST RESUME: Skip if already downloaded
            if location_id in existing_ids:
                print(f"⏭️ Skipping already downloaded: {location_name}")
                continue
            
            # This is a NEW station
            station_num = completed_count + total_new_stations + 1
            print(f"\n📍 Processing NEW station #{station_num}: {location_name}")
            
            # Get sensors
            sensors = location.get("sensors", [])
            station_data = []
            
            # Collect pollution data
            for sensor in sensors:
                param_name = sensor.get("parameter", {}).get("name")
                
                if param_name in REQUIRED_POLLUTION:
                    sensor_id = sensor.get("id")
                    measurements = fetch_all_measurements(sensor_id, param_name)
                    
                    if measurements:
                        station_data.extend(measurements)
                        print(f"  ✓ {param_name}: {len(measurements)} records")
            
            # Get weather data
            coords = location.get("coordinates", {})
            weather_data = fetch_weather_data(
                coords.get("latitude"), 
                coords.get("longitude")
            )
            
            if weather_data:
                station_data.extend(weather_data)
                print(f"  ✓ Weather: {len(weather_data)} records")
            
            # Add to current batch
            if station_data:
                df_station = pd.DataFrame(station_data)
                df_station['station_id'] = location_id
                df_station['station_name'] = location_name
                df_station['country'] = country
                df_station['latitude'] = coords.get('latitude')
                df_station['longitude'] = coords.get('longitude')
                
                current_batch.append(df_station)
                stations_in_batch += 1
                total_new_stations += 1
                
                # ✅ PROFESSIONAL APPEND: Save batch when reach 10 or last station
                if stations_in_batch == 10 or total_new_stations == remaining:
                    # Combine current batch
                    batch_df = pd.concat(current_batch, ignore_index=True)
                    
                    # ✅ TRUE APPEND MODE - no rewriting!
                    batch_df.to_csv(
                        OUTPUT_FILE,
                        mode='a',           # Append mode
                        header=not file_exists,  # Write header only for first batch
                        index=False
                    )
                    
                    # After first batch, file exists
                    file_exists = True
                    
                    print(f"\n📊 CHECKPOINT: Saved batch of {stations_in_batch} stations")
                    print(f"📁 Appended {len(batch_df):,} new rows to {OUTPUT_FILE}")
                    print(f"🎯 Total stations now: {completed_count + total_new_stations}")
                    
                    # Clear batch for next group
                    current_batch = []
                    stations_in_batch = 0
            
            time.sleep(1)  # Rate limiting
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ COLLECTION COMPLETE!")
    print(f"📊 Total stations in {OUTPUT_FILE}: {completed_count + total_new_stations}")
    print("=" * 70)

# ==============================
# EXECUTE
# ==============================

if __name__ == "__main__":
    start_time = time.time()
    
    # Run the optimized collection
    collect_remaining_stations()
    
    # Final row count (memory efficient)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            final_rows = sum(1 for line in f) - 1
        print(f"\n📈 Final row count: {final_rows:,}")
    
    print(f"\n⏱️  Total time: {round(time.time() - start_time, 2)} seconds")