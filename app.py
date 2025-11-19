import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import os
import warnings
import reverse_geocoder as rg  # <--- NEW: Converts Lat/Lon to City Name
import verification_scraper_module as scraper_module

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(page_title="FloodWatch AI", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00BFFF;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 FloodWatch AI: Early Warning System")
st.markdown("### 🛰️ Satellite-Based Cloudburst & Flood Prediction System")
st.write("Fusion of **NOAA GFS Satellite Data** (Science) and **Live News Signals** (Verification).")
st.divider()

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_brain():
    try:
        bst = xgb.Booster()
        bst.load_model('xgb_flood_model.json')
        with open('model_columns.json', 'r') as f:
            cols = json.load(f)
        return bst, cols
    except Exception as e:
        return None, None

bst, model_cols = load_ai_brain()

if not bst:
    st.error("❌ CRITICAL ERROR: Model files not found. Please upload 'xgb_flood_model.json' and 'model_columns.json' to GitHub.")
    st.stop()

# --- HELPER: CALCULATE GFS DATE ---
def get_gfs_date():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    cycle = 18 if hour >= 22 else 12 if hour >= 16 else 6 if hour >= 10 else 0
    date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if cycle == 18 and hour < 4: date -= timedelta(days=1)
    return date

# --- HELPER: SINGLE POINT FETCH (For Tab 1) ---
def fetch_satellite_data(lat, lon):
    status_container = st.empty()
    status_container.info(f"📡 Connecting to NOAA Satellite Grid for {lat}, {lon}...")
    
    try:
        date = get_gfs_date()
        H = Herbie(date=date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
        variables = {
            'dpt': 'DPT:2 m above ground', 'lhtfl': 'LHTFL:surface', 
            'shtfl': 'SHTFL:surface', 'tcdc': 'TCDC:entire atmosphere', 
            'clwmr': 'CLWMR:entire atmosphere', 'ugrd': 'UGRD:10 m above ground', 
            'vgrd': 'VGRD:10 m above ground'
        }
        
        data = {'lat': lat, 'lon': lon}
        for var, search in variables.items():
            try:
                ds = H.xarray(f":{search}:")
                if isinstance(ds, list): ds = ds[0]
                val = ds.sel(latitude=lat, longitude=lon, method='nearest')
                data[var] = float(list(val.data_vars.values())[0].values)
            except:
                data[var] = 0.0
                
        status_container.success("✅ Satellite Data Acquired!")
        return data
    except Exception as e:
        status_container.error(f"❌ Satellite Link Failed: {e}")
        return None

# --- HELPER: MASSIVE GRID FETCH (For Tab 2) ---
def fetch_india_grid():
    status = st.empty()
    status.info("📡 Downloading Full India Grid (This takes ~30 seconds)...")
    
    try:
        date = get_gfs_date()
        H = Herbie(date=date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
        variables = {
            'dpt': 'DPT:2 m above ground', 'lhtfl': 'LHTFL:surface', 
            'shtfl': 'SHTFL:surface', 'tcdc': 'TCDC:entire atmosphere', 
            'clwmr': 'CLWMR:entire atmosphere', 'ugrd': 'UGRD:10 m above ground', 
            'vgrd': 'VGRD:10 m above ground'
        }
        
        all_data = {}
        coords = None
        bar = st.progress(0)
        
        for i, (var, search) in enumerate(variables.items()):
            try:
                ds = H.xarray(f":{search}:")
                if isinstance(ds, list): ds = ds[0]
                
                # Slice for India (Lat 37 to 6, Lon 68 to 98)
                ds_india = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
                
                # Save Coords once
                if coords is None:
                    lats = ds_india.latitude.values
                    lons = ds_india.longitude.values
                    # Handle 1D vs 2D coordinates
                    if lats.ndim == 1:
                        lon_grid, lat_grid = np.meshgrid(lons, lats)
                    else:
                        lon_grid, lat_grid = lons, lats
                    coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
                
                # Flatten data
                val_key = list(ds_india.data_vars.keys())[0]
                all_data[var] = ds_india[val_key].values.flatten()
                
            except:
                pass # Skip missing vars
            
            bar.progress((i+1)/len(variables))
            
        if coords:
            df = pd.DataFrame(coords)
            for v, d in all_data.items():
                if v not in df.columns:
                    if d is not None and len(d) == len(df): df[v] = d
                    else: df[v] = 0.0
            
            status.success(f"✅ Grid Scan Complete! ({len(df)} points analyzed)")
            return df
        return None
    except Exception as e:
        status.error(f"Grid Scan Failed: {e}")
        return None

# --- PREDICTION FUNCTIONS ---
def predict_single_risk(data_dict):
    df = pd.DataFrame([data_dict])
    df = df.rename(columns={'dpt':'dew2m', 'lhtfl':'latent_flux', 'shtfl':'sensible_flux', 'tcdc':'cloud_cover', 'clwmr':'cloud_liquid'})
    
    df['latent_flux'] *= -3600
    df['sensible_flux'] *= -3600
    df['cloud_cover'] /= 100.0
    df['wind_speed'] = np.sqrt(data_dict.get('ugrd',0)**2 + data_dict.get('vgrd',0)**2)
    df['month'] = datetime.now().month
    df['year'] = datetime.now().year
    
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
        
    return float(bst.predict(xgb.DMatrix(df[model_cols]))[0])

def predict_grid_risk(df_raw):
    """Vectorized prediction for thousands of points at once"""
    df = df_raw.copy()
    df = df.rename(columns={'dpt':'dew2m', 'lhtfl':'latent_flux', 'shtfl':'sensible_flux', 'tcdc':'cloud_cover', 'clwmr':'cloud_liquid'})
    
    # Vectorized Math (Fast)
    if 'latent_flux' in df.columns: df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: df['cloud_cover'] /= 100.0
    
    if 'ugrd' in df.columns and 'vgrd' in df.columns:
        df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    else:
        df['wind_speed'] = 0.0
        
    df['month'] = datetime.now().month
    df['year'] = datetime.now().year
    
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
        
    # Predict
    dmat = xgb.DMatrix(df[model_cols])
    df['risk_score'] = bst.predict(dmat)
    return df

# --- TABS ---
tab1, tab2 = st.tabs(["📍 Check Specific Location", "🚨 Auto-Scan High Risk Zones"])

# --- TAB 1: MANUAL CHECK (UNCHANGED) ---
with tab1:
    st.subheader("Inspect a Specific Location")
    c1, c2, c3 = st.columns([1, 1, 2])
    input_lat = c1.number_input("Latitude", value=13.08, format="%.4f")
    input_lon = c2.number_input("Longitude", value=80.27, format="%.4f")
    city_name = c3.text_input("City Name (for News Verification)", "Chennai")
    
    if st.button("🔍 Run Analysis", type="primary"):
        data = fetch_satellite_data(input_lat, input_lon)
        
        if data:
            # 1. Science Score
            science_risk = predict_single_risk(data)
            
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🛰️ Satellite Probability", f"{science_risk:.1%}", help="Based on GFS Physics")
            
            # 2. News Score
            with st.spinner(f"📰 Verifying with News Reports for {city_name}..."):
                scraper_module.clear_cache_for_new_location()
                scraper = scraper_module.VerificationScraper(location=city_name, hours_lookback=24)
                news_data = scraper.scrape_all()
                
                text_score = 0.0
                if not news_data.empty:
                    text_score = 0.95 
                    col_b.metric("📰 News Signals", f"{len(news_data)} Articles", delta="Confirmed")
                    with st.expander("View Source Articles"):
                        st.dataframe(news_data[['title', 'published_date', 'source']])
                else:
                    col_b.metric("📰 News Signals", "0 Articles", delta="Silent", delta_color="off")

            # 3. Fusion Decision
            final_score = (science_risk * 0.8) + (text_score * 0.2)
            col_c.metric("🔥 Final Risk Confidence", f"{final_score:.1%}")
            
            st.divider()
            if final_score > 0.75:
                st.error(f"🚨 DISASTER ALERT ISSUED FOR {city_name.upper()}")
                st.write("**Action:** Immediate evacuation warning recommended.")
            elif final_score > 0.40:
                st.warning(f"⚠️ CAUTION: Unstable Weather in {city_name.upper()}")
            else:
                st.success(f"✅ SAFE: No imminent threat for {city_name.upper()}")

# --- TAB 2: AUTO SCAN (NEW GRID LOGIC) ---
with tab2:
    st.subheader("🤖 Autonomous National Scanner")
    st.write("Scanning **15,000+ Grid Points** across India to find the absolute highest risk zones.")
    
    if st.button("Start Satellite Scan"):
        
        # 1. Download & Predict
        grid_df = fetch_india_grid()
        
        if grid_df is not None:
            st.write("Analyzing Atmospheric Physics...")
            results_df = predict_grid_risk(grid_df)
            
            # 2. Get Top 10 Riskiest Points
            top_risks = results_df.nlargest(10, 'risk_score').copy()
            
            # 3. Convert Coords to City Names (Reverse Geocoding)
            st.write("Identifying Locations...")
            coords = list(zip(top_risks['lat'], top_risks['lon']))
            geo_results = rg.search(coords)
            
            top_risks['Location'] = [f"{x['name']}, {x['admin1']}" for x in geo_results]
            
            # 4. Display Results Table
            display_data = []
            for idx, row in top_risks.iterrows():
                risk = row['risk_score']
                status = "SAFE"
                if risk > 0.75: status = "🚨 DANGER"
                elif risk > 0.40: status = "⚠️ CAUTION"
                
                display_data.append({
                    "Location": row['Location'],
                    "Risk %": f"{risk:.1%}",
                    "Raw Score": risk,
                    "Status": status,
                    "Lat/Lon": f"{row['lat']:.2f}, {row['lon']:.2f}"
                })
            
            df_display = pd.DataFrame(display_data)
            st.table(df_display[["Location", "Risk %", "Status", "Lat/Lon"]])
            
            # Highlight Worst
            worst = df_display.iloc[0]
            if worst['Raw Score'] > 0.75:
                st.error(f"🚨 PRIORITY ALERT: {worst['Location']} is at {worst['Risk %']} Risk!")
            else:
                st.success("✅ National Scan: No major threats detected.")
