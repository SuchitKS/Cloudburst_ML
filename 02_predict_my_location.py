import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import warnings
import os

warnings.filterwarnings('ignore')

# --- 👤 USER CONFIGURATION ---
MY_LAT = 9.5000
MY_LON = 73.0000
LOCATION_NAME = "Kavaratti"

print("="*60)
print(f"📍 SMART CLOUDBURST MONITOR: {LOCATION_NAME.upper()}")
print("="*60)

def fetch_and_scan():
    # 1. Setup GFS Cycle
    print("🔍 Finding latest GFS forecast...")
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    if hour >= 22: cycle = 18
    elif hour >= 16: cycle = 12
    elif hour >= 10: cycle = 6
    else: cycle = 0
    
    forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if cycle == 18 and hour < 4: forecast_date -= timedelta(days=1)
    
    print(f"   Target Cycle: {forecast_date.strftime('%Y-%m-%d %H:00')} UTC")

    # 2. Connect to Herbie
    try:
        H = Herbie(
            date=forecast_date, model='gfs', product='pgrb2.0p25', 
            fxx=6, save_dir='herbie_cache'
        )
    except Exception as e:
        print(f"❌ GFS Connection Failed: {e}")
        return

    # 3. Define Scanning Box (0.5 deg ~ 50km radius)
    min_lat, max_lat = MY_LAT - 0.5, MY_LAT + 0.5
    min_lon, max_lon = MY_LON - 0.5, MY_LON + 0.5

    print("📥 Scanning 50km Radius for Storms...")
    
    # Variables (Added Precip 'apcp' as the Veto)
    variables = {
        'dpt':   ':DPT:2 m above', 
        'lhtfl': ':LHTFL:surface', 
        'shtfl': ':SHTFL:surface', 
        'tcdc':  ':TCDC:entire atmosphere', 
        'cwat':  ':CWAT:', 
        'ugrd':  ':UGRD:10 m above', 
        'vgrd':  ':VGRD:10 m above',
        'apcp':  ':APCP:surface'  # <--- NEW: Total Precipitation (Rain)
    }
    
    all_data = {}
    coords = None
    
    for var, search in variables.items():
        try:
            ds = H.xarray(search)
            if isinstance(ds, list): ds = ds[0]
            
            # Select the 50km box
            ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            
            val_key = list(ds.data_vars.keys())[0]
            all_data[var] = ds[val_key].values.flatten()
            
            if coords is None:
                lats = ds.latitude.values
                lons = ds.longitude.values
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
        except:
            # APCP is critical, if missing assume 0 (No Rain)
            if var == 'apcp': all_data['apcp'] = None
            pass

    if not all_data:
        print("❌ No data found for this region.")
        return

    # 4. Create Dataframe
    df = pd.DataFrame(coords)
    for v, d in all_data.items(): 
        if d is not None: df[v] = d
    
    # Fill missing APCP with 0
    if 'apcp' not in df.columns: df['apcp'] = 0.0

    # 5. Format Data (Physics Mode)
    rename_map = {
        'dpt': 'dew2m', 'lhtfl': 'latent_flux', 'shtfl': 'sensible_flux', 
        'tcdc': 'cloud_cover', 'cwat': 'cloud_liquid'
    }
    df = df.rename(columns=rename_map)
    
    # Unit Corrections
    if 'latent_flux' in df.columns: df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: df['cloud_cover'] /= 100.0
    if 'ugrd' in df.columns: df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    
    # Fill missing model columns
    if not os.path.exists('model_columns.json'):
        print("❌ Error: model_columns.json not found.")
        return

    with open('model_columns.json', 'r') as f:
        cols = json.load(f)
    
    for c in cols: 
        if c not in df.columns: df[c] = 0.0
        
    # 6. Load Model & Predict
    if not os.path.exists('xgb_flood_model.json'):
        print("❌ Error: xgb_flood_model.json not found.")
        return

    bst = xgb.Booster()
    bst.load_model('xgb_flood_model.json')
    
    dmatrix = xgb.DMatrix(df[cols])
    probs = bst.predict(dmatrix)
    df['probability'] = probs

    # 7. SMART LOGIC: The Precipitation Veto
    # Find the point with the highest AI Risk
    best_idx = df['probability'].idxmax()
    max_prob = df['probability'].max()
    
    # Check if it is ACTUALLY raining there (Physics Check)
    local_rain = df.iloc[best_idx].get('apcp', 0.0)
    
    print("-" * 50)
    print(f"☔ Local Rain:  {local_rain:.2f} mm")
    print(f"🌊 Raw AI Risk: {max_prob:.4f}")
    
    final_prob = max_prob
    
    # RULE: If Rain < 0.5mm, it CANNOT be a cloudburst.
    # This kills "Ghost Storms" (Fog/Dew) instantly.
    if local_rain < 0.5 and max_prob > 0.5:
        print("🛑 VETO: Satellite sees Zero Rain. Suppressing False Alarm.")
        final_prob = 0.0000
    
    print(f"🎯 FINAL PROB:  {final_prob:.4f}")
    print("-" * 50)

    # Save to JSON
    info = {
        "location_name": LOCATION_NAME,
        "lat": MY_LAT,
        "lon": MY_LON,
        "probability": float(final_prob),
        "timestamp": str(datetime.now())
    }
    with open('my_location_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✅ Saved smart prediction for {LOCATION_NAME}.")

if __name__ == "__main__":
    fetch_and_scan()