import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import warnings
import os

warnings.filterwarnings('ignore')

print("="*60)
print("🌊 GLOBAL CLOUDBURST MAP GENERATOR (SMART MODE)")
print("="*60)

# --- 1. FETCH GFS DATA (With Rain & Temp Checks) ---
def fetch_latest_gfs_data():
    print("🔍 Finding latest GFS forecast...")
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    if hour >= 22: cycle = 18
    elif hour >= 16: cycle = 12
    elif hour >= 10: cycle = 6
    else: cycle = 0
    
    forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if cycle == 18 and hour < 4: forecast_date -= timedelta(days=1)
    
    print(f"🎯 Cycle: {forecast_date.strftime('%Y-%m-%d %H:00')} UTC")
    
    try:
        H = Herbie(date=forecast_date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
    except:
        return None

    # Added 'apcp' (Rain) and 'tmp' (Temperature) for validity checks
    variables = {
        'dpt': ':DPT:2 m above', 
        'lhtfl': ':LHTFL:surface',
        'shtfl': ':SHTFL:surface', 
        'tcdc': ':TCDC:entire atmosphere',
        'cwat': ':CWAT:', 
        'ugrd': ':UGRD:10 m above',
        'vgrd': ':VGRD:10 m above',
        'apcp': ':APCP:surface',      # <--- VETO CHECK 1 (Rain)
        'tmp':  ':TMP:2 m above'      # <--- VETO CHECK 2 (Cold/Fog)
    }
    
    print(f"📥 Downloading Map Data...")
    all_data = {}
    coords = None
    
    for v, s in variables.items():
        try:
            ds = H.xarray(s)
            if isinstance(ds, list): ds = ds[0]
            # Crop to Indian Subcontinent
            ds = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
            
            # Dynamic variable name handling
            val_key = list(ds.data_vars.keys())[0]
            all_data[v] = ds[val_key].values.flatten()
            
            if coords is None:
                lats = ds.latitude.values
                lons = ds.longitude.values
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
        except: 
            # If Rain/Temp is missing, fill with safe defaults
            if v == 'apcp': all_data[v] = np.zeros_like(list(all_data.values())[0])
            pass
        
    if coords is None: return None
    df = pd.DataFrame(coords)
    for v, d in all_data.items(): 
        if d is not None: df[v] = d
        
    # Fill missing APCP/TMP
    if 'apcp' not in df.columns: df['apcp'] = 0.0
    if 'tmp' not in df.columns: df['tmp'] = 290.0 # Default warm
        
    return df

# --- 2. FORMAT DATA & APPLY LOGIC ---
def process_and_predict(df_raw):
    print("🔧 Processing Physics & Applying Logic...")
    df = df_raw.copy()
    
    # 1. Rename to Model Features
    rename_map = {'dpt': 'dew2m', 'lhtfl': 'latent_flux', 'shtfl': 'sensible_flux', 
                  'tcdc': 'cloud_cover', 'cwat': 'cloud_liquid'}
    df = df.rename(columns=rename_map)
    
    # 2. Unit Corrections
    if 'latent_flux' in df.columns: df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: df['cloud_cover'] /= 100.0
    if 'ugrd' in df.columns: df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    
    # 3. Prepare Model Input (Strictly 8 features)
    if not os.path.exists('model_columns.json'):
        print("❌ Error: model_columns.json missing")
        return None

    with open('model_columns.json', 'r') as f:
        model_cols = json.load(f)
        
    # Fill missing model cols
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
        
    # 4. PREDICT
    if not os.path.exists('xgb_flood_model.json'):
        print("❌ Error: xgb_flood_model.json missing")
        return None

    bst = xgb.Booster()
    bst.load_model('xgb_flood_model.json')
    
    dmatrix = xgb.DMatrix(df[model_cols])
    probs = bst.predict(dmatrix)
    df['probability'] = probs
    
    # --- 5. THE VETO LOGIC (Crucial Step) ---
    print("🛑 Applying 'Smart Filters' (Rain & Temperature)...")
    
    # Count high risks before filtering
    initial_risks = len(df[df['probability'] > 0.8])
    
    # A. PRECIPITATION VETO
    # If Rain < 0.2mm, force Probability to 0.0
    # (The model might see humidity, but if satellite says "No Rain", we trust satellite)
    df.loc[df['apcp'] < 0.2, 'probability'] = 0.0
    
    # B. TEMPERATURE VETO
    # If Temp < 10°C (283.15 K), force Probability to 0.0
    # (Prevents Fog/Dew being mistaken for Cloudburst)
    df.loc[df['tmp'] < 283.15, 'probability'] = 0.0
    
    final_risks = len(df[df['probability'] > 0.8])
    print(f"   • Filtered False Alarms: {initial_risks - final_risks} locations removed.")
    print(f"   • Valid Storm Locations: {final_risks}")

    # 6. Select Output Columns
    output_df = df[['lat', 'lon', 'probability', 'apcp', 'tmp']]
    
    return output_df

def run_prediction():
    df_raw = fetch_latest_gfs_data()
    if df_raw is None: return
    
    df_final = process_and_predict(df_raw)
    if df_final is None: return
    
    # Load Threshold
    try:
        with open('training_metadata.json', 'r') as f:
            thresh = json.load(f)['optimal_threshold']
    except: thresh = 0.85
    
    df_final['prediction'] = (df_final['probability'] >= thresh).astype(int)
    
    print(f"\n📊 Max Probability Detected: {df_final['probability'].max():.4f}")
    
    # Show Top Risks
    top_risks = df_final.nlargest(5, 'probability')
    if top_risks.iloc[0]['probability'] > 0.0:
        print(f"🎯 Top Verified Risks:")
        print(top_risks[['lat', 'lon', 'probability', 'apcp']].to_string(index=False))
    else:
        print("✅ No Cloudbursts detected over India right now.")
    
    df_final.to_csv('science_predictions.csv', index=False)
    print("\n✅ Saved to science_predictions.csv")

if __name__ == "__main__":
    run_prediction()