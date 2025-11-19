import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import os
import warnings
import verification_scraper_module as scraper_module
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_lottie import st_lottie
import requests

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION (Dark Mode & Wide Layout) ---
st.set_page_config(page_title="FloodWatch AI", page_icon="🌊", layout="wide")

# Custom CSS for "Command Center" Look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #00BFFF;
        text-shadow: 0 0 10px #00BFFF;
    }
    .stMetric {
        background-color: #1f2630;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .css-1d391kg {
        background-color: #1f2630;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: LOAD LOTTIE ANIMATION ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Load Satellite Animation
lottie_satellite = load_lottieurl("https://lottie.host/63e9b042-967f-4aa9-8137-3c55ba533082/C2t4Qk7sFf.json")

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
    st.error("❌ CRITICAL ERROR: Model files missing. Please upload 'xgb_flood_model.json' to GitHub.")
    st.stop()

# --- VISUALIZATION FUNCTIONS ---
def create_gauge_chart(score):
    """Creates a cool speedometer chart for Risk Score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cloudburst Probability"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00BFFF"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "#00FF00"},  # Green
                {'range': [40, 70], 'color': "#FFA500"}, # Orange
                {'range': [70, 100], 'color': "#FF0000"} # Red
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig

def fetch_satellite_data(lat, lon):
    # Only show status if not in auto-scan mode to avoid clutter
    status = st.empty()
    
    try:
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        cycle = 18 if hour >= 22 else 12 if hour >= 16 else 6 if hour >= 10 else 0
        forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
        if cycle == 18 and hour < 4: forecast_date -= timedelta(days=1)
        
        H = Herbie(date=forecast_date, model='gfs', product='pgrb2.0p25', fxx=6, save_dir='herbie_cache')
        
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
                
        return data
        
    except Exception as e:
        return None

def predict_risk(data_dict):
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

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    if lottie_satellite:
        st_lottie(lottie_satellite, height=150, key="sidebar_anim")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=100) # Static fallback
    
    app_mode = st.selectbox("Select Mode", ["📍 Single Location Check", "🚨 National Auto-Scan"])
    st.divider()
    st.caption("Powered by XGBoost & NOAA GFS")

# --- MODE 1: SINGLE LOCATION ---
if app_mode == "📍 Single Location Check":
    st.title("📍 Precision Risk Analyzer")
    
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude", value=13.08, format="%.4f")
    lon = c2.number_input("Longitude", value=80.27, format="%.4f")
    city = c3.text_input("City Name", "Chennai")
    
    if st.button("🚀 Launch Satellite Scan", type="primary"):
        with st.spinner("🛰️ Establishing Uplink with NOAA Satellite..."):
            data = fetch_satellite_data(lat, lon)
        
        if data:
            # Risk Calculation
            risk = predict_risk(data)
            
            # Layout: Gauge Chart Left, Metrics Right
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.plotly_chart(create_gauge_chart(risk), use_container_width=True)
            
            with col_right:
                st.subheader("Atmospheric Telemetry")
                m1, m2 = st.columns(2)
                m1.metric("Humidity (Dew)", f"{data['dpt']:.1f} K")
                m2.metric("Wind Speed", f"{np.sqrt(data.get('ugrd',0)**2 + data.get('vgrd',0)**2):.1f} m/s")
                
                st.divider()
                
                # NEWS CHECK
                st.subheader("📰 News Verification Signal")
                scraper_module.clear_cache_for_new_location()
                scraper = scraper_module.VerificationScraper(location=city, hours_lookback=24)
                news = scraper.scrape_all()
                
                if not news.empty:
                    st.warning(f"⚠️ {len(news)} Disaster Reports Found")
                    with st.expander("Read Reports"):
                        st.dataframe(news[['title', 'source']])
                    final_score = (risk * 0.8) + (0.95 * 0.2)
                else:
                    st.success("✅ No Panic Signals in News")
                    final_score = (risk * 0.8) + (0.0 * 0.2)
            
            # Final Verdict Banner
            st.divider()
            if final_score > 0.75:
                st.error(f"🚨 CRITICAL ALERT: CLOUDBURST IMMINENT IN {city.upper()} (Confidence: {final_score:.1%})")
            elif final_score > 0.4:
                st.warning(f"⚠️ CAUTION: Unstable Weather Detected in {city.upper()}")
            else:
                st.success(f"✅ STATUS: SAFE. No threat detected for {city.upper()}")

# --- MODE 2: NATIONAL SCAN ---
elif app_mode == "🚨 National Auto-Scan":
    st.title("🛡️ National Threat Scanner")
    st.markdown("Scanning strategic high-value targets across the Indian Subcontinent.")
    
    if st.button("📡 Initialize Grid Scan"):
        # Map Setup
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB dark_matter")
        
        zones = [
            {"City": "Chennai", "Lat": 13.08, "Lon": 80.27},
            {"City": "Mumbai", "Lat": 19.07, "Lon": 72.87},
            {"City": "Delhi", "Lat": 28.61, "Lon": 77.20},
            {"City": "Bangalore", "Lat": 12.97, "Lon": 77.59},
            {"City": "Kolkata", "Lat": 22.57, "Lon": 88.36},
            {"City": "Kedarnath", "Lat": 30.73, "Lon": 79.07},
            {"City": "Nicobar", "Lat": 6.00, "Lon": 92.50},
            {"City": "Cochin", "Lat": 9.93, "Lon": 76.26},
            {"City": "Hyderabad", "Lat": 17.38, "Lon": 78.48}
        ]
        
        progress_bar = st.progress(0)
        status_log = st.empty()
        
        results_table = []
        
        for i, z in enumerate(zones):
            status_log.code(f"Scanning Sector: {z['City']} [{z['Lat']}, {z['Lon']}] ...")
            d = fetch_satellite_data(z['Lat'], z['Lon'])
            
            if d:
                risk = predict_risk(d)
                color = "green"
                if risk > 0.75: color = "red"
                elif risk > 0.40: color = "orange"
                
                # Add to Folium Map
                folium.CircleMarker(
                    location=[z['Lat'], z['Lon']],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    popup=f"{z['City']}: {risk:.1%} Risk"
                ).add_to(m)
                
                results_table.append({"Location": z['City'], "Risk Score": f"{risk:.1%}", "Status": "🚨 DANGER" if risk > 0.75 else "✅ SAFE"})
            
            progress_bar.progress((i+1)/len(zones))
            
        status_log.code("Scan Complete. Rendering Tactical Map...")
        
        # Layout: Map on Left, Data on Right
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st_folium(m, width=700, height=500)
            
        with c2:
            st.subheader("Threat Log")
            df_res = pd.DataFrame(results_table)
            st.dataframe(df_res, hide_index=True)

