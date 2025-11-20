import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import datetime, timedelta
from herbie import Herbie
import warnings
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch
from transformers import pipeline

# Import your verification scraper
try:
    import verification_scraper_module as scraper_module
except ImportError:
    st.error("⚠️ verification_scraper_module.py not found in the project directory!")

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌊 Cloudburst Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #2980b9, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .danger-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .safe-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_xgboost_model():
    """Load the XGBoost model"""
    try:
        if os.path.exists('xgb_flood_model.json'):
            bst = xgb.Booster()
            bst.load_model('xgb_flood_model.json')
            return bst
        else:
            st.error("❌ xgb_flood_model.json not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_model_columns():
    """Load model column names"""
    try:
        if os.path.exists('model_columns.json'):
            with open('model_columns.json', 'r') as f:
                return json.load(f)
        else:
            st.error("❌ model_columns.json not found!")
            return None
    except Exception as e:
        st.error(f"Error loading model columns: {e}")
        return None

def fetch_gfs_data_for_location(lat, lon, radius=0.5):
    """Fetch GFS data for a specific location"""
    try:
        # Determine latest GFS cycle
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        if hour >= 22: cycle = 18
        elif hour >= 16: cycle = 12
        elif hour >= 10: cycle = 6
        else: cycle = 0
        
        forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
        if cycle == 18 and hour < 4: 
            forecast_date -= timedelta(days=1)
        
        # Connect to Herbie
        H = Herbie(
            date=forecast_date, 
            model='gfs', 
            product='pgrb2.0p25', 
            fxx=6, 
            save_dir='herbie_cache'
        )
        
        # Define box around location
        min_lat, max_lat = lat - radius, lat + radius
        min_lon, max_lon = lon - radius, lon + radius
        
        # Variables to fetch
        variables = {
            'dpt': ':DPT:2 m above', 
            'lhtfl': ':LHTFL:surface', 
            'shtfl': ':SHTFL:surface', 
            'tcdc': ':TCDC:entire atmosphere', 
            'cwat': ':CWAT:', 
            'ugrd': ':UGRD:10 m above', 
            'vgrd': ':VGRD:10 m above',
            'apcp': ':APCP:surface',
            'tmp': ':TMP:2 m above'
        }
        
        all_data = {}
        coords = None
        
        for var, search in variables.items():
            try:
                ds = H.xarray(search)
                if isinstance(ds, list): ds = ds[0]
                
                ds = ds.sel(
                    latitude=slice(max_lat, min_lat), 
                    longitude=slice(min_lon, max_lon)
                )
                
                val_key = list(ds.data_vars.keys())[0]
                all_data[var] = ds[val_key].values.flatten()
                
                if coords is None:
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
            except:
                if var in ['apcp', 'tmp']:
                    all_data[var] = None
                pass
        
        if not all_data or coords is None:
            return None
        
        df = pd.DataFrame(coords)
        for v, d in all_data.items(): 
            if d is not None: 
                df[v] = d
        
        # Fill missing values
        if 'apcp' not in df.columns: df['apcp'] = 0.0
        if 'tmp' not in df.columns: df['tmp'] = 290.0
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching GFS data: {e}")
        return None

def process_data_for_prediction(df_raw):
    """Process raw data for model prediction"""
    df = df_raw.copy()
    
    # Rename columns
    rename_map = {
        'dpt': 'dew2m', 
        'lhtfl': 'latent_flux', 
        'shtfl': 'sensible_flux', 
        'tcdc': 'cloud_cover', 
        'cwat': 'cloud_liquid'
    }
    df = df.rename(columns=rename_map)
    
    # Unit corrections
    if 'latent_flux' in df.columns: 
        df['latent_flux'] *= -3600
    if 'sensible_flux' in df.columns: 
        df['sensible_flux'] *= -3600
    if 'cloud_cover' in df.columns: 
        df['cloud_cover'] /= 100.0
    if 'ugrd' in df.columns and 'vgrd' in df.columns:
        df['wind_speed'] = np.sqrt(df['ugrd']**2 + df['vgrd']**2)
    
    return df

def predict_cloudburst_location(lat, lon, location_name):
    """Main prediction function for a specific location"""
    model = load_xgboost_model()
    model_cols = load_model_columns()
    
    if model is None or model_cols is None:
        return None
    
    # Fetch data
    with st.spinner('📡 Fetching satellite data...'):
        df_raw = fetch_gfs_data_for_location(lat, lon)
    
    if df_raw is None:
        st.error("Failed to fetch GFS data")
        return None
    
    # Process data
    df = process_data_for_prediction(df_raw)
    
    # Fill missing model columns
    for c in model_cols:
        if c not in df.columns:
            df[c] = 0.0
    
    # Predict
    dmatrix = xgb.DMatrix(df[model_cols])
    probs = model.predict(dmatrix)
    df['probability'] = probs
    
    # Get best prediction
    best_idx = df['probability'].idxmax()
    max_prob = df['probability'].max()
    local_rain = df.iloc[best_idx].get('apcp', 0.0)
    local_temp = df.iloc[best_idx].get('tmp', 290.0)
    
    # Apply veto logic
    final_prob = max_prob
    veto_applied = False
    veto_reason = ""
    
    if local_rain < 0.5 and max_prob > 0.5:
        final_prob = 0.0
        veto_applied = True
        veto_reason = "No precipitation detected by satellite"
    
    if local_temp < 283.15 and max_prob > 0.5:
        final_prob = 0.0
        veto_applied = True
        veto_reason = "Temperature too cold (likely fog, not rain)"
    
    result = {
        'location_name': location_name,
        'lat': lat,
        'lon': lon,
        'raw_probability': float(max_prob),
        'final_probability': float(final_prob),
        'local_rain': float(local_rain),
        'local_temp': float(local_temp),
        'veto_applied': veto_applied,
        'veto_reason': veto_reason,
        'timestamp': datetime.now().isoformat()
    }
    
    return result

def run_verification(location_name):
    """Run news verification for a location"""
    try:
        scraper_module.clear_cache_for_new_location()
        
        scraper = scraper_module.VerificationScraper(
            location=location_name,
            hours_lookback=48,
            enable_fallback=True,
            min_articles_threshold=5
        )
        
        data = scraper.scrape_all()
        
        if data.empty:
            return 0.0, 0, []
        
        # Simple keyword-based scoring
        danger_keywords = [
            'flood', 'cloudburst', 'heavy rain', 'downpour', 
            'waterlogging', 'evacuation', 'alert', 'warning'
        ]
        
        data['text_blob'] = data['title'].fillna('') + " " + data['summary'].fillna('')
        
        scores = []
        top_articles = []
        
        for idx, row in data.iterrows():
            text = row['text_blob'].lower()
            keyword_count = sum(1 for kw in danger_keywords if kw in text)
            
            if keyword_count >= 2:
                score = min(keyword_count / len(danger_keywords), 1.0)
                scores.append(score)
                top_articles.append({
                    'title': row['title'],
                    'source': row['source'],
                    'score': score
                })
        
        text_score = np.mean(scores) if scores else 0.0
        
        return float(text_score), len(data), sorted(top_articles, key=lambda x: x['score'], reverse=True)[:5]
        
    except Exception as e:
        st.error(f"Verification error: {e}")
        return 0.0, 0, []

def fetch_india_wide_data():
    """Fetch GFS data for entire India"""
    try:
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        if hour >= 22: cycle = 18
        elif hour >= 16: cycle = 12
        elif hour >= 10: cycle = 6
        else: cycle = 0
        
        forecast_date = now_utc.replace(hour=cycle, minute=0, second=0, microsecond=0)
        if cycle == 18 and hour < 4:
            forecast_date -= timedelta(days=1)
        
        H = Herbie(
            date=forecast_date, 
            model='gfs', 
            product='pgrb2.0p25', 
            fxx=6, 
            save_dir='herbie_cache'
        )
        
        variables = {
            'dpt': ':DPT:2 m above', 
            'lhtfl': ':LHTFL:surface',
            'shtfl': ':SHTFL:surface', 
            'tcdc': ':TCDC:entire atmosphere',
            'cwat': ':CWAT:', 
            'ugrd': ':UGRD:10 m above',
            'vgrd': ':VGRD:10 m above',
            'apcp': ':APCP:surface',
            'tmp': ':TMP:2 m above'
        }
        
        all_data = {}
        coords = None
        
        for v, s in variables.items():
            try:
                ds = H.xarray(s)
                if isinstance(ds, list): ds = ds[0]
                
                # Crop to Indian subcontinent
                ds = ds.sel(latitude=slice(37, 6), longitude=slice(68, 98))
                
                val_key = list(ds.data_vars.keys())[0]
                all_data[v] = ds[val_key].values.flatten()
                
                if coords is None:
                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    coords = {'lat': lat_grid.flatten(), 'lon': lon_grid.flatten()}
            except:
                if v in ['apcp', 'tmp']:
                    all_data[v] = np.zeros_like(list(all_data.values())[0]) if all_data else None
                pass
        
        if coords is None:
            return None
            
        df = pd.DataFrame(coords)
        for v, d in all_data.items():
            if d is not None:
                df[v] = d
        
        if 'apcp' not in df.columns: df['apcp'] = 0.0
        if 'tmp' not in df.columns: df['tmp'] = 290.0
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching India-wide data: {e}")
        return None

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🌊 Cloudburst Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Early Warning System for India</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["📍 Location Check", "🗺️ India Scan"])
    
    # TAB 1: Location Check
    with tab1:
        st.header("🎯 Check Specific Location")
        st.write("Enter coordinates to check cloudburst risk for your area")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_name = st.text_input("📌 Location Name", value="Kavaratti", help="Name of the place")
        
        with col2:
            lat = st.number_input("🌐 Latitude", value=9.5000, format="%.4f", help="Latitude coordinate")
        
        with col3:
            lon = st.number_input("🌐 Longitude", value=73.0000, format="%.4f", help="Longitude coordinate")
        
        if st.button("🚀 Analyze Location", key="analyze_btn"):
            st.markdown("---")
            
            # Step 1: Science Prediction
            st.subheader("🛰️ Step 1: Satellite & Physics Analysis")
            result = predict_cloudburst_location(lat, lon, location_name)
            
            if result:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🌧️ Rain Detected", f"{result['local_rain']:.2f} mm")
                
                with col2:
                    st.metric("🌡️ Temperature", f"{result['local_temp']-273.15:.1f}°C")
                
                with col3:
                    st.metric("🤖 AI Risk Score", f"{result['raw_probability']:.2%}")
                
                with col4:
                    st.metric("🎯 Final Score", f"{result['final_probability']:.2%}")
                
                if result['veto_applied']:
                    st.warning(f"🛡️ Veto Applied: {result['veto_reason']}")
                
                science_prob = result['final_probability']
                
                # Step 2: News Verification
                st.markdown("---")
                st.subheader("📰 Step 2: News Verification")
                
                with st.spinner('🔍 Scanning news sources...'):
                    text_score, article_count, top_articles = run_verification(location_name)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📄 Articles Found", article_count)
                
                with col2:
                    st.metric("📊 Text Signal Score", f"{text_score:.2%}")
                
                if top_articles:
                    st.write("**Top Confirming Articles:**")
                    for article in top_articles:
                        st.write(f"- [{article['score']:.0%}] {article['title']} - *{article['source']}*")
                
                # Step 3: Fusion Decision
                st.markdown("---")
                st.subheader("🧠 Step 3: Final Fusion Decision")
                
                SCIENCE_WEIGHT = 0.70
                TEXT_WEIGHT = 0.30
                THRESHOLD = 0.75
                
                final_score = (science_prob * SCIENCE_WEIGHT) + (text_score * TEXT_WEIGHT)
                
                # Display gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = final_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Level", 'font': {'size': 24}},
                    delta = {'reference': THRESHOLD * 100, 'increasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#00f2fe'},
                            {'range': [50, 75], 'color': '#ffd89b'},
                            {'range': [75, 100], 'color': '#f5576c'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': THRESHOLD * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision message
                if final_score >= THRESHOLD:
                    st.markdown(f"""
                    <div class="danger-card">
                        <h2>🚨 DANGER: HIGH RISK OF CLOUDBURST</h2>
                        <h3>Confidence: {final_score:.1%}</h3>
                        <p><strong>ACTION:</strong> Monitor local emergency channels immediately. Consider evacuation if advised.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif final_score >= 0.5:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); padding: 20px; border-radius: 10px; color: white;">
                        <h2>⚠️ CAUTION: CONDITIONS ARE UNSTABLE</h2>
                        <h3>Confidence: {final_score:.1%}</h3>
                        <p><strong>ACTION:</strong> Stay alert, but no immediate confirmation of disaster.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-card">
                        <h2>✅ SAFE: NO IMMINENT THREAT DETECTED</h2>
                        <h3>Confidence: {final_score:.1%}</h3>
                        <p><strong>STATUS:</strong> Normal weather conditions.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 2: India Scan
    with tab2:
        st.header("🗺️ India-Wide Cloudburst Scan")
        st.write("Scan entire India to identify the 10 most vulnerable locations")
        
        if st.button("🔍 Start India Scan", key="scan_btn"):
            model = load_xgboost_model()
            model_cols = load_model_columns()
            
            if model is None or model_cols is None:
                st.error("Model files not found!")
                return
            
            st.markdown("---")
            
            # Fetch India-wide data
            with st.spinner('🛰️ Downloading satellite data for India...'):
                df_raw = fetch_india_wide_data()
            
            if df_raw is None:
                st.error("Failed to fetch data")
                return
            
            # Process data
            with st.spinner('⚙️ Processing atmospheric data...'):
                df = process_data_for_prediction(df_raw)
                
                # Fill missing columns
                for c in model_cols:
                    if c not in df.columns:
                        df[c] = 0.0
                
                # Predict
                dmatrix = xgb.DMatrix(df[model_cols])
                probs = model.predict(dmatrix)
                df['probability'] = probs
                
                # Apply veto filters
                initial_risks = len(df[df['probability'] > 0.8])
                
                df.loc[df['apcp'] < 0.2, 'probability'] = 0.0
                df.loc[df['tmp'] < 283.15, 'probability'] = 0.0
                
                final_risks = len(df[df['probability'] > 0.8])
            
            st.success(f"✅ Filtered {initial_risks - final_risks} false alarms")
            st.info(f"📍 {final_risks} valid storm locations identified")
            
            # Get top 10 locations
            top_10 = df.nlargest(10, 'probability')[['lat', 'lon', 'probability', 'apcp', 'tmp']]
            
            if top_10.iloc[0]['probability'] > 0.0:
                st.subheader("🎯 Top 10 High-Risk Locations")
                
                # Create map
                fig = px.scatter_mapbox(
                    top_10,
                    lat='lat',
                    lon='lon',
                    size='probability',
                    color='probability',
                    color_continuous_scale='Reds',
                    size_max=30,
                    zoom=4,
                    mapbox_style='open-street-map',
                    hover_data={'probability': ':.2%', 'apcp': ':.2f', 'tmp': ':.1f'}
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.subheader("📊 Detailed Risk Table")
                display_df = top_10.copy()
                display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
                display_df['apcp'] = display_df['apcp'].apply(lambda x: f"{x:.2f} mm")
                display_df['tmp'] = display_df['tmp'].apply(lambda x: f"{x-273.15:.1f}°C")
                display_df.columns = ['Latitude', 'Longitude', 'Risk Score', 'Rainfall', 'Temperature']
                st.dataframe(display_df, use_container_width=True)
                
            else:
                st.success("✅ No significant cloudburst threats detected over India right now.")

if __name__ == "__main__":
    main()