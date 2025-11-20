import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import os
import warnings
import reverse_geocoder as rg
import folium
from streamlit_folium import st_folium
import verification_scraper_module as scraper_module

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(page_title="FloodWatch AI", page_icon="🌊", layout="wide")

st.title("🌊 FloodWatch AI: Early Warning System")
st.markdown("### 🛰️ Satellite-Based Cloudburst & Flood Prediction System")
st.write("Fusion of **NOAA GFS Satellite Data** (Science) and **Live News Signals** (Verification).")
st.divider()

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_brain():
    try:
        bst = xgb.Booster()
        bst.load_model('xgb_cloudburst_model.json')
        with open('model_columns.json', 'r') as f:
            cols = json.load(f)
        return bst, cols
    except Exception as e:
        return None, None

bst, model_cols = load_ai_brain()

if not bst:
    st.error("❌ CRITICAL ERROR: Model files not found. Please upload 'xgb_cloudburst_model.json' and 'model_columns.json'.")
    st.stop()

# --- DATA FETCHING ---

def get_gfs_date():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    cycle = 18 if hour >= 22 else 12 if hour >= 16 else 6 if hour >= 10 else 0
    date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if cycle == 18 and hour < 4: date -= timedelta(days=1)
    return date

def fetch_single_point(lat, lon):
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
        for v, search in variables.items():
            try:
                ds = H.xarray(f":{search}:")
                if isinstance(ds, list): ds = ds[0]
                val = ds.sel(latitude=lat, longitude=lon, method='nearest')
                data[v] = float(list(val.data_vars.values())[0].values)
            except: data[v] = 0.0
        return data
    except: return None

def fetch_india_grid():
    status = st.empty()
    status.info("📡 Downloading Full India Grid (15,000+ points). This takes ~30 seconds...")
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
                ds_india = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
                if coords is None:
                    lats = ds_india.latitude.values
                    lons = ds_india.longitude.values
                    if lats.ndim == 1: lon_grid, lat_grid = np.meshgrid(lons, lats)
                    else: lon_grid, lat_grid = lons, lats
                    coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
                val_key = list(ds_india.data_vars.keys())[0]
                all_data[var] = ds_india[val_key].values.flatten()
            except: pass
            bar.progress((i+1)/len(variables))
        if coords:
            df = pd.DataFrame(coords)
            for v, d in all_data.items():
                if v not in df.columns:
                    if d is not None and len(d) == len(df): df[v] = d
                    else: df[v] = 0.0
            status.success(f"✅ Grid Scan Complete! Analyzed {len(df)} locations.")
            return df
        return None
    except Exception as e:
        status.error(f"Grid Scan Failed: {e}")
        return None

# --- PREDICTION LOGIC ---
def process_data_and_predict(df_raw):
    df = df_raw.copy()
    df = df.rename(columns={'dpt':'dew2m', 'lhtfl':'latent_flux', 'shtfl':'sensible_flux', 'tcdc':'cloud_cover', 'clwmr':'cloud_liquid'})
    if 'latent_flux' in df.columns: df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: df['cloud_cover'] /= 100.0
    if 'ugrd' in df.columns and 'vgrd' in df.columns:
        df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    else: df['wind_speed'] = 0.0
    df['month'] = datetime.now().month
    df['year'] = datetime.now().year
    for c in model_cols:
        if c not in df.columns: df[c] = 0.0
    dmat = xgb.DMatrix(df[model_cols])
    df['risk_score'] = bst.predict(dmat)
    return df

# --- TABS ---
tab1, tab2 = st.tabs(["📍 Check Specific Location", "🚨 Auto-Scan High Risk Zones"])

# --- TAB 1: MANUAL CHECK ---
with tab1:
    st.subheader("Inspect a Specific Location")
    c1, c2, c3 = st.columns([1, 1, 2])
    input_lat = c1.number_input("Latitude", value=13.08, format="%.4f")
    input_lon = c2.number_input("Longitude", value=80.27, format="%.4f")
    city_name = c3.text_input("City Name", "Chennai")
    
    if st.button("🔍 Run Analysis", type="primary"):
        data = fetch_single_point(input_lat, input_lon)
        if data:
            df = pd.DataFrame([data])
            res = process_data_and_predict(df)
            risk = res['risk_score'].iloc[0]
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🛰️ Satellite Probability", f"{risk:.1%}")
            with st.spinner(f"📰 Checking News for {city_name}..."):
                scraper_module.clear_cache_for_new_location()
                scraper = scraper_module.VerificationScraper(location=city_name, hours_lookback=24)
                news_data = scraper.scrape_all()
                text_score = 0.0
                if not news_data.empty:
                    text_score = 0.95 
                    col_b.metric("📰 News Signals", f"{len(news_data)} Articles", delta="Confirmed")
                    with st.expander("View Articles"): st.dataframe(news_data[['title', 'published_date']])
                else: col_b.metric("📰 News Signals", "0 Articles", delta="Silent", delta_color="off")
            final_score = (risk * 0.8) + (text_score * 0.2)
            col_c.metric("🔥 Final Confidence", f"{final_score:.1%}")
            st.divider()
            if final_score > 0.75: st.error(f"🚨 DISASTER ALERT ISSUED FOR {city_name.upper()}")
            elif final_score > 0.40: st.warning(f"⚠️ CAUTION: Unstable Weather in {city_name.upper()}")
            else: st.success(f"✅ SAFE: No imminent threat for {city_name.upper()}")

# --- TAB 2: AUTO SCAN (FIXED WITH SESSION STATE) ---
with tab2:
    st.subheader("🤖 Autonomous National Scanner")
    st.write("Scanning **15,000+ Grid Points** across India.")
    
    # Initialize Session State to store results
    if 'scan_data' not in st.session_state:
        st.session_state['scan_data'] = None

    # Run Scan ONLY when button is clicked
    if st.button("🚀 INITIATE SATELLITE SCAN", type="primary"):
        grid_df = fetch_india_grid()
        
        if grid_df is not None:
            st.write("Analyzing Physics & Risks...")
            results_df = process_data_and_predict(grid_df)
            st.write("Filtering Locations...")
            
            candidates = results_df.nlargest(500, 'risk_score').copy()
            coords = list(zip(candidates['lat'], candidates['lon']))
            geo_results = rg.search(coords)
            
            candidates['City'] = [x['name'] for x in geo_results]
            candidates['State'] = [x['admin1'] for x in geo_results]
            candidates['Country'] = [x['cc'] for x in geo_results]
            
            india_risks = candidates[candidates['Country'] == 'IN'].head(10)
            if india_risks.empty: india_risks = candidates.head(5)
            
            # Save to Session State (Memory)
            st.session_state['scan_data'] = india_risks
    
    # Display Results (from Memory)
    if st.session_state['scan_data'] is not None:
        india_risks = st.session_state['scan_data']
        
        m = folium.Map(location=[20.59, 78.96], zoom_start=5, tiles="CartoDB dark_matter")
        
        map_data = []
        for _, row in india_risks.iterrows():
            risk_pct = row['risk_score'] * 100
            color = "red" if risk_pct > 75 else "orange"
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8, color=color, fill=True, fill_color=color,
                popup=f"{row['City']}: {risk_pct:.1f}%"
            ).add_to(m)
            
            map_data.append({
                "Location": f"{row['City']}, {row['State']}",
                "Risk %": f"{risk_pct:.1f}%",
                "Status": "🚨 DANGER" if risk_pct > 75 else "⚠️ CAUTION"
            })
        
        st_folium(m, width=800, height=500)
        st.subheader("🚨 Top Risk Zones Detected")
        st.table(pd.DataFrame(map_data))
        
        if st.button("Clear Results"):
            st.session_state['scan_data'] = None
            st.rerun()



