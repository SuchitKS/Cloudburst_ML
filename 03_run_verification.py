import json
import numpy as np
import torch
from transformers import pipeline
from datetime import datetime
import warnings
import os

# Import your scraper module
try:
    import verification_scraper_module as scraper_module
except ImportError:
    print("❌ Error: 'verification_scraper_module.py' not found.")
    exit()

warnings.filterwarnings('ignore')

print("="*60)
print("🗞️  STEP 3: NEWS VERIFICATION AGENT")
print("="*60)

# --- 1. CONFIGURATION ---
# Based on your Step 2 results (Lat 25.00, Lon 72.50)
TARGET_LOCATION = "Rajasthan"   # <--- CHANGE THIS to the city/state you want to check
HOURS_LOOKBACK = 48             # Look back 48 hours

print(f"🔎 Target Location: {TARGET_LOCATION}")
print(f"⏱️  Lookback:        {HOURS_LOOKBACK} hours")

# --- 2. Load NLP Model ---
print("\n🧠 Loading AI Text Analyzer...")
try:
    # Use a zero-shot classifier (classifies text without specific training)
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    exit()

# Labels to look for
labels = ["flood disaster", "heavy rain", "cloudburst", "waterlogging", "normal weather"]

# --- 3. Run Web Scraper ---
print("\n🌐 Scraping News & Social Media...")

# Clear old cache to ensure fresh news
scraper_module.clear_cache_for_new_location()

scraper = scraper_module.VerificationScraper(
    location=TARGET_LOCATION,
    hours_lookback=HOURS_LOOKBACK,
    enable_fallback=True
)

# Get the data
df = scraper.scrape_all()

# --- 4. Calculate Signal Score ---
text_signal_score = 0.0
top_articles = []

if not df.empty:
    print(f"\n📊 Analyzing {len(df)} articles...")
    
    # Combine Title + Summary for analysis
    df['text'] = df['title'] + " " + df['summary']
    texts = df['text'].tolist()
    
    scores = []
    for i, txt in enumerate(texts):
        if len(txt) < 20: continue
        try:
            # Ask AI: "Is this text about a flood/rain disaster?"
            res = classifier(txt[:512], candidate_labels=labels, multi_label=False)
            
            # Get score of 'flood', 'heavy rain', 'cloudburst'
            risk_score = sum([res['scores'][res['labels'].index(L)] for L in labels[:3]])
            scores.append(risk_score)
            
            if risk_score > 0.5:
                top_articles.append({'title': df.iloc[i]['title'], 'score': risk_score})
                
        except:
            pass
            
    if scores:
        # Final score is the average of the risk detected
        text_signal_score = float(np.mean(scores))
else:
    print("⚠️  No recent news found (Score = 0.0)")

# --- 5. Save Results ---
print("-" * 40)
print(f"📈 FINAL TEXT SIGNAL SCORE: {text_signal_score:.4f}")
print("-" * 40)

if top_articles:
    print("🚨 Top Confirming Articles:")
    for a in top_articles[:3]:
        print(f"   • [{a['score']:.2f}] {a['title']}")

output = {
    'location': TARGET_LOCATION,
    'text_signal_score': text_signal_score,
    'articles_found': len(df),
    'timestamp': str(datetime.now())
}

with open('text_signal_score.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to 'text_signal_score.json'")
print("👉 Next: Run '04_run_fusion_alert.py' to combine Science + News.")