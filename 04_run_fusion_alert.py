import pandas as pd
import json
import os
from datetime import datetime

print("\n" + "="*60)
print("🚀 FINAL PHASE: FUSION & ALERT GENERATION")
print("="*60)

# --- 1. CONFIGURATION ---
# 70% Science (Physics is primary) + 30% News (Verification is secondary)
SCIENCE_WEIGHT = 0.70  
TEXT_WEIGHT    = 0.30  
ALERT_THRESHOLD = 0.75 

# --- 2. LOAD SCIENCE DATA (Step 2) ---
science_file = 'science_predictions.csv'

try:
    if not os.path.exists(science_file):
        raise FileNotFoundError("Run Step 2 first!")

    science_df = pd.read_csv(science_file)
    
    if science_df.empty:
        print("⚠️  Science data is empty.")
        max_science_score = 0.0
        science_lat = 0.0
        science_lon = 0.0
    else:
        # Find the location with the highest risk
        top_row = science_df.loc[science_df['probability'].idxmax()]
        max_science_score = float(top_row['probability'])
        science_lat = float(top_row['lat'])
        science_lon = float(top_row['lon'])
        
        print(f"✅ Science Model (Physics):")
        print(f"   • Max Probability: {max_science_score:.2%} (Weight: {SCIENCE_WEIGHT})")
        print(f"   • Epicenter:       Lat {science_lat:.2f}, Lon {science_lon:.2f}")

except Exception as e:
    print(f"❌ Error loading science data: {e}")
    exit()

# --- 3. LOAD NEWS DATA (Step 3) ---
text_file = 'text_signal_score.json'

try:
    if not os.path.exists(text_file):
        print("⚠️  Verification file not found. Assuming 0 news signal.")
        text_signal_score = 0.0
        location_name = "Unknown"
    else:
        with open(text_file, 'r') as f:
            text_data = json.load(f)
        
        text_signal_score = float(text_data.get('text_signal_score', 0.0))
        location_name = text_data.get('location', 'Unknown')
        articles = text_data.get('articles_found', 0)
        
        print(f"\n✅ Verification Agent (News):")
        print(f"   • Signal Score:    {text_signal_score:.2%} (Weight: {TEXT_WEIGHT})")
        print(f"   • Location:        {location_name}")
        print(f"   • Articles Found:  {articles}")

except Exception as e:
    print(f"❌ Error loading verification data: {e}")
    text_signal_score = 0.0

# --- 4. FUSION CALCULATION ---
print("\n" + "-"*60)
print("🧮 CALCULATING FUSION SCORE")
print("-" * 60)

# The Math: (Science * 0.70) + (News * 0.30)
weighted_science = max_science_score * SCIENCE_WEIGHT
weighted_text = text_signal_score * TEXT_WEIGHT
final_confidence = weighted_science + weighted_text

print(f"   Science Contribution:  {weighted_science:.4f}")
print(f"   News Contribution:     {weighted_text:.4f}")
print(f"   --------------------------------------------------")
print(f"   FINAL FUSION SCORE:    {final_confidence:.4f} / 1.0")

# --- 5. DECISION ---
print("\n" + "="*60)
print("📢 SYSTEM DECISION")
print("="*60)

if final_confidence >= ALERT_THRESHOLD:
    print("\n🚨🚨🚨  CRITICAL DISASTER ALERT  🚨🚨🚨")
    print(f"   CONFIDENCE: {final_confidence:.1%}")
    print(f"   LOCATION:   {location_name.upper()} (Lat: {science_lat}, Lon: {science_lon})")
    print("   ACTION:     IMMEDIATE EVACUATION / WARNING RECOMMENDED")
    
elif final_confidence >= 0.5:
    print("\n⚠️  WATCH WARNING (ORANGE ALERT)")
    print(f"   CONFIDENCE: {final_confidence:.1%}")
    print(f"   LOCATION:   {location_name.upper()}")
    print("   ACTION:     Monitor situation. High risk detected.")

else:
    print("\n✅ NO IMMEDIATE THREAT")
    print(f"   CONFIDENCE: {final_confidence:.1%}")
    print("   STATUS:     Below alert threshold.")

print("="*60 + "\n")