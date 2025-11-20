import json
import os

print("="*60)
print("🧠 PERSONAL FUSION DECISION")
print("="*60)

# --- Configuration ---
WT_SCIENCE = 0.70  # 70% Weight to Satellite/Physics
WT_TEXT    = 0.30  # 30% Weight to News
THRESHOLD  = 0.75  # Alert Threshold

try:
    if not os.path.exists('my_location_info.json'):
        raise FileNotFoundError
        
    with open('my_location_info.json', 'r') as f:
        info = json.load(f)
        
    loc = info.get('location_name', 'Unknown')
    
    # Get scores (default to 0 if missing)
    sci_prob = info.get('probability', 0.0)
    text_score = info.get('text_score', 0.0)
    articles = info.get('articles_found', 0)
    timestamp = info.get('timestamp', 'Unknown')
    
    # Calculate Fusion
    final_score = (sci_prob * WT_SCIENCE) + (text_score * WT_TEXT)
    
    print(f"📍 Location:      {loc}")
    print(f"🕒 Timestamp:     {timestamp}")
    print(f"📰 Articles:      {articles}")
    print("-" * 40)
    print(f"🛰️  Science Prob:  {sci_prob:.4f} (x {WT_SCIENCE})")
    print(f"📰 News Signal:   {text_score:.4f} (x {WT_TEXT})")
    print("-" * 40)
    print(f"🔥 FINAL SCORE:   {final_score:.4f}")
    print("-" * 40)
    
    # Decision Logic
    if final_score >= THRESHOLD:
        print("\n🚨🚨🚨 DANGER: HIGH RISK OF CLOUDBURST 🚨🚨🚨")
        print(f"Confidence: {final_score:.2%}")
        print("ACTION: Monitor local emergency channels immediately.")
        
    elif final_score >= 0.5:
        print("\n⚠️  CAUTION: CONDITIONS ARE UNSTABLE")
        print(f"Confidence: {final_score:.2%}")
        print("ACTION: Stay alert, but no immediate confirmation of disaster.")
        
    else:
        print("\n✅ SAFE: NO IMMINENT THREAT DETECTED")
        print(f"Confidence: {final_score:.2%}")
        
except FileNotFoundError:
    print("❌ Error: 'my_location_info.json' not found.")
    print("   Run Step 2 and Step 3 first.")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)