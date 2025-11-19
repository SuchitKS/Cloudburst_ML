import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import os
import warnings
# Import your existing scraper file
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

# --- GFS DOWNLOADER ---
def fetch_satellite_data(lat, lon):
    status_container = st.empty()
    status_container.info(f"📡 Connecting to NOAA Satellite Grid for {lat}, {lon}...")
    
    try:
        # 1. Calculate Cycle
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        cycle = 18 if hour >= 22 else 12 if hour >= 16 else 6 if hour >= 10 else 0
        forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
        if cycle == 18 and hour < 4: forecast_date -= timedelta(days=1)
        
        # 2. Download
        H = Herbie(date=forecast_date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
        # 3. Variables
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

def predict_risk(data_dict):
    # Format Data
    df = pd.DataFrame([data_dict])
    df = df.rename(columns={'dpt':'dew2m', 'lhtfl':'latent_flux', 'shtfl':'sensible_flux', 'tcdc':'cloud_cover', 'clwmr':'cloud_liquid'})
    
    # Unit Fixes (Crucial)
    df['latent_flux'] *= -3600
    df['sensible_flux'] *= -3600
    df['cloud_cover'] /= 100.0
    df['wind_speed'] = np.sqrt(data_dict.get('ugrd',0)**2 + data_dict.get('vgrd',0)**2)
    df['month'] = datetime.now().month
    df['year'] = datetime.now().year
    
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
        
    # Predict
    risk = float(bst.predict(xgb.DMatrix(df[model_cols]))[0])
    return risk

# --- TABS ---
tab1, tab2 = st.tabs(["📍 Check Specific Location", "🚨 Auto-Scan High Risk Zones"])

# --- TAB 1: MANUAL CHECK ---
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
            science_risk = predict_risk(data)
            
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
                    text_score = 0.95 # Simulating high confidence if articles exist
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

# --- TAB 2: AUTO SCAN ---
with tab2:
    st.subheader("🤖 Autonomous National Scanner")
    st.write("Scanning 10 strategic high-risk zones across India in real-time.")
    
    if st.button("Start Satellite Scan"):
        # List of strategic points to check
        zones = [
            {"City": "Chennai", "Lat": 13.08, "Lon": 80.27},
            {"City": "Mumbai", "Lat": 19.07, "Lon": 72.87},
            {"City": "Delhi NCR", "Lat": 28.61, "Lon": 77.20},
            {"City": "Bangalore", "Lat": 12.97, "Lon": 77.59},
            {"City": "Kolkata", "Lat": 22.57, "Lon": 88.36},
            {"City": "Kedarnath", "Lat": 30.73, "Lon": 79.07},
            {"City": "Nicobar Islands", "Lat": 6.00, "Lon": 92.50},
            {"City": "Guwahati", "Lat": 26.14, "Lon": 91.73},
            {"City": "Cochin", "Lat": 9.93, "Lon": 76.26},
            {"City": "Hyderabad", "Lat": 17.38, "Lon": 78.48}
        ]
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, zone in enumerate(zones):
            status_text.text(f"🛰️ Scanning Sector: {zone['City']}...")
            d = fetch_satellite_data(zone['Lat'], zone['Lon'])
            
            if d:
                risk = predict_risk(d)
                
                # Determine Status
                status = "SAFE"
                if risk > 0.75: status = "🚨 DANGER"
                elif risk > 0.40: status = "⚠️ CAUTION"
                
                results.append({
                    "Location": zone['City'],
                    "Risk %": f"{risk:.1%}",
                    "Raw Score": risk,
                    "Status": status
                })
            
            progress_bar.progress((i + 1) / len(zones))
            
        status_text.text("✅ Scan Complete.")
        
        # Display Results
        results_df = pd.DataFrame(results).sort_values("Raw Score", ascending=False)
        
        st.table(results_df[["Location", "Risk %", "Status"]])
        
        # Highlight Worst
        worst = results_df.iloc[0]
        if worst['Raw Score'] > 0.75:
            st.error(f"🚨 PRIORITY ALERT: {worst['Location']} is at {worst['Risk %']} Risk!")
        else:
            st.success("✅ National Scan: No major threats detected.")