import json
import numpy as np
import torch
from transformers import pipeline
import warnings
import os
import sys

# Import your scraper module
try:
    import verification_scraper_module as scraper_module
except ImportError:
    print("❌ Error: 'verification_scraper_module.py' not found.")
    exit()

warnings.filterwarnings('ignore')

print("="*60)
print("🗞️  PERSONAL VERIFICATION CHECKER (ROBUST MODE)")
print("="*60)

# --- 1. Load Target Info ---
try:
    with open('my_location_info.json', 'r') as f:
        info = json.load(f)
    target_city = info['location_name']
    print(f"📍 Target Location: {target_city}")
except FileNotFoundError:
    print("❌ Error: 'my_location_info.json' not found.")
    print("   Please run '02_predict_my_location.py' first.")
    exit()

# --- 2. Load NLP Model (With Fallback) ---
print("🧠 Loading Text Analysis Model...")
device = 0 if torch.cuda.is_available() else -1
classifier = None
model_status = "AI"

try:
    # Try loading the Zero-Shot model
    classifier = pipeline(
        "zero-shot-classification", 
        model="valhalla/distilbart-mnli-12-3",
        device=device
    )
    print("✅ AI Model loaded successfully.")
except Exception as e:
    print(f"⚠️  RAM Limit Reached: {e}")
    print("🔄 Switching to KEYWORD MODE (Low Memory)...")
    model_status = "KEYWORD"

disaster_labels = [
    "flood disaster", "heavy rainfall", "water logging", 
    "emergency evacuation", "weather warning"
]

# --- 3. Scrape News ---
print(f"🔎 Scraping news for '{target_city}'...")
scraper_module.clear_cache_for_new_location()

# Initialize scraper
scraper = scraper_module.VerificationScraper(
    location=target_city, 
    hours_lookback=24,
    enable_fallback=True
)
data = scraper.scrape_all()

# --- 4. Analyze Articles ---
text_score = 0.0
article_count = len(data)

def analyze_with_keywords(text):
    """Fallback function if AI crashes"""
    # List of danger words
    danger_words = [
        'flood', 'inundation', 'heavy rain', 'downpour', 'cloudburst',
        'cyclone', 'storm', 'evacuation', 'alert', 'warning', 'red alert',
        'orange alert', 'waterlogging', 'drowning', 'relief camp'
    ]
    
    text_lower = text.lower()
    score = 0.0
    found_words = 0
    
    for word in danger_words:
        if word in text_lower:
            found_words += 1
            
    # Simple scoring logic
    if found_words >= 3: score = 0.9
    elif found_words == 2: score = 0.6
    elif found_words == 1: score = 0.3
    
    return score

if not data.empty:
    print(f"📊 Analyzing {article_count} articles using {model_status} mode...")
    
    # Combine Title + Summary
    data['text_blob'] = data['title'].fillna('') + " " + data['summary'].fillna('')
    texts = data['text_blob'].tolist()
    
    scores = []
    for txt in texts:
        if len(txt) < 10: continue
        
        try:
            if classifier:
                # AI MODE
                res = classifier(txt[:512], candidate_labels=disaster_labels, multi_label=False)
                # Take average confidence of top 3 disaster labels
                top_scores = sorted(res['scores'], reverse=True)[:3]
                avg_conf = np.mean(top_scores)
                scores.append(avg_conf)
            else:
                # KEYWORD MODE
                kw_score = analyze_with_keywords(txt)
                scores.append(kw_score)
        except:
            pass
    
    if scores:
        # Text score is the average confidence of all relevant articles
        text_score = float(np.mean(scores))
else:
    print("   No articles found (Score = 0.0)")

print("-" * 40)
print(f"📰 TEXT VERIFICATION SCORE: {text_score:.4f}")
print("-" * 40)

# --- 5. Save Results ---
info['text_score'] = text_score
info['articles_found'] = article_count
info['analysis_mode'] = model_status

with open('my_location_info.json', 'w') as f:
    json.dump(info, f, indent=2)
    
print("✅ Verification complete. Run Step 4 next.")