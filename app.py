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
import verification_scraper_module as scraper_module
from streamlit_lottie import st_lottie
import requests
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(page_title="FloodWatch AI", page_icon="🌊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1 { color: #00BFFF; text-align: center; }
    .stMetric { background-color: #1f2630; border: 1px solid #30363d; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Animation
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_sat = load_lottieurl("https://lottie.host/63e9b042-967f-4aa9-8137-3c55ba533082/C2t4Qk7sFf.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_brain():
    try:
        bst = xgb.Booster()
        bst.load_model('xgb_flood_model.json')
        with open('model_columns.json', 'r') as f:
            cols = json.load(f)
        return bst, cols
    except: return None, None

bst, model_cols = load_brain()

if not bst:
    st.error("❌ Model Missing! Upload xgb_flood_model.json")
    st.stop()

# --- DATA FETCHERS ---
def get_gfs_date():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    cycle = 18 if hour >= 22 else 12 if hour >= 16 else 6 if hour >= 10 else 0
    date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
    if cycle == 18 and hour < 4: date -= timedelta(days=1)
    return date

def fetch_point_data(lat, lon):
    """Downloads single point for Manual Check"""
    try:
        date = get_gfs_date()
        H = Herbie(date=date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
        vars_map = {
            'dpt': 'DPT:2 m above ground', 'lhtfl': 'LHTFL:surface', 
            'shtfl': 'SHTFL:surface', 'tcdc': 'TCDC:entire atmosphere', 
            'clwmr': 'CLWMR:entire atmosphere', 'ugrd': 'UGRD:10 m above ground', 
            'vgrd': 'VGRD:10 m above ground'
        }
        
        data = {'lat': lat, 'lon': lon}
        for v, s in vars_map.items():
            try:
                ds = H.xarray(f":{s}:")
                if isinstance(ds, list): ds = ds[0]
                val = ds.sel(latitude=lat, longitude=lon, method='nearest')
                data[v] = float(list(val.data_vars.values())[0].values)
            except: data[v] = 0.0
        return data
    except: return None

def fetch_india_grid():
    """Downloads FULL INDIA GRID for Auto-Scan"""
    status = st.empty()
    status.info("📡 Downloading Full India Grid (This takes ~30 seconds)...")
    
    try:
        date = get_gfs_date()
        H = Herbie(date=date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
        vars_map = {
            'dpt': 'DPT:2 m above ground', 'lhtfl': 'LHTFL:surface', 
            'shtfl': 'SHTFL:surface', 'tcdc': 'TCDC:entire atmosphere', 
            'clwmr': 'CLWMR:entire atmosphere', 'ugrd': 'UGRD:10 m above ground', 
            'vgrd': 'VGRD:10 m above ground'
        }
        
        all_data = {}
        coords = None
        bar = st.progress(0)
        
        for i, (v, s) in enumerate(vars_map.items()):
            try:
                ds = H.xarray(f":{s}:")
                if isinstance(ds, list): ds = ds[0]
                ds_india = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
                
                if coords is None:
                    lats = ds_india.latitude.values
                    lons = ds_india.longitude.values
                    if lats.ndim == 1: lon_grid, lat_grid = np.meshgrid(lons, lats)
                    else: lon_grid, lat_grid = lons, lats
                    coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
                
                val_key = list(ds_india.data_vars.keys())[0]
                all_data[v] = ds_india[val_key].values.flatten()
            except: pass
            bar.progress((i+1)/len(vars_map))
            
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

# --- DASHBOARD UI ---
with st.sidebar:
    if lottie_sat: st_lottie(lottie_sat, height=150)
    else: st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=100)
    
    st.header("⚙️ Control Panel")
    mode = st.radio("Select Operation Mode:", ["📍 Manual Check", "🛰️ Full India Auto-Scan"])

# --- TAB 1: MANUAL CHECK ---
if mode == "📍 Manual Check":
    st.title("📍 Precision Risk Analyzer")
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Lat", value=13.08)
    lon = c2.number_input("Lon", value=80.27)
    city = c3.text_input("City", "Chennai")
    
    if st.button("Run Analysis", type="primary"):
        d = fetch_point_data(lat, lon)
        if d:
            risk = predict_single_risk(d)
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk*100,
                title={'text': "Cloudburst Probability"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00BFFF"},
                       'steps': [{'range': [0,40], 'color': "#00FF00"}, {'range': [70,100], 'color': "#FF0000"}]}
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            
            c1, c2 = st.columns([1, 1])
            with c1: st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("News Verification")
                scraper_module.clear_cache_for_new_location()
                scraper = scraper_module.VerificationScraper(location=city)
                news = scraper.scrape_all()
                if not news.empty:
                    st.warning(f"{len(news)} Articles Found")
                    st.dataframe(news[['title']])
                else: st.success("No News Panic Signals")

# --- TAB 2: AUTO SCAN (WITH INDIA FILTER) ---
elif mode == "🛰️ Full India Auto-Scan":
    st.title("🛡️ National Autonomous Scanner")
    st.markdown("Scanning **15,000+ Grid Points**. Filtering for **INDIA ONLY**.")
    
    if st.button("🚀 INITIATE SATELLITE SCAN", type="primary"):
        
        grid_df = fetch_india_grid()
        
        if grid_df is not None:
            st.write("Analyzing Atmospheric Physics...")
            results_df = predict_grid_risk(grid_df)
            
            st.write("Filtering for Indian Territory...")
            # 1. Take Top 500 Risky Points (Candidate Pool)
            candidates = results_df.nlargest(500, 'risk_score').copy()
            
            # 2. Reverse Geocode
            coords = list(zip(candidates['lat'], candidates['lon']))
            geo_results = rg.search(coords)
            
            candidates['City'] = [x['name'] for x in geo_results]
            candidates['State'] = [x['admin1'] for x in geo_results]
            candidates['Country'] = [x['cc'] for x in geo_results]
            
            # 3. FILTER: KEEP ONLY INDIA ('IN')
            india_only = candidates[candidates['Country'] == 'IN'].head(10)
            
            if india_only.empty:
                st.warning("No high risks found inside Indian borders. Showing closest neighbors:")
                india_only = candidates.head(5) # Fallback
            
            # 4. Map & Table
            m = folium.Map(location=[20.59, 78.96], zoom_start=5, tiles="CartoDB dark_matter")
            
            map_data = []
            for _, row in india_only.iterrows():
                risk_pct = row['risk_score'] * 100
                color = "red" if risk_pct > 75 else "orange"
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8, color=color, fill=True, fill_color=color,
                    popup=f"{row['City']}: {risk_pct:.1f}%"
                ).add_to(m)
                
                map_data.append({
                    "City": row['City'],
                    "State": row['State'],
                    "Risk %": f"{risk_pct:.1f}%",
                    "Lat/Lon": f"{row['lat']:.2f}, {row['lon']:.2f}"
                })
            
            c1, c2 = st.columns([2, 1])
            with c1: st_folium(m, width=700, height=500)
            with c2: 
                st.subheader("🚨 Top 10 Indian Risk Zones")
                st.dataframe(pd.DataFrame(map_data))
