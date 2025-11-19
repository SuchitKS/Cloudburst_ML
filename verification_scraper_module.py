# ========================================================================
# VERIFICATION PHASE - Flood/Cloudburst Fusion System
# Cross-check XGBoost predictions with human signals (News + Reddit)
# ✨ NEW: Parallel scraping + Automatic state-level fallback
# ========================================================================

# Step 1: Import necessary libraries
import feedparser
import pandas as pd
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ✨ NEW: Parallel processing imports
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# DistilBERT imports for text classification
from transformers import pipeline
import torch
import numpy as np

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"🖥️  Using device: {'GPU' if device == 0 else 'CPU'}\n")

# ========================================================================
# CACHE MANAGEMENT - Clear old cache when location changes
# ========================================================================

def clear_cache_for_new_location(cache_file="news_cache.pkl"):
    """Clear cache file to ensure fresh data for new location."""
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"🗑️  Cache cleared: {cache_file}\n")
            return True
        except Exception as e:
            print(f"⚠️  Could not clear cache: {e}\n")
            return False
    return True

# ========================================================================
# ✨ NEW: LOCATION HIERARCHY FOR FALLBACK
# ========================================================================

LOCATION_HIERARCHY = {
    # Karnataka
    'bengaluru': 'karnataka',
    'bangalore': 'karnataka',
    'mysore': 'karnataka',
    'mysuru': 'karnataka',
    'mangalore': 'karnataka',
    'mangaluru': 'karnataka',
    'hubli': 'karnataka',
    'belgaum': 'karnataka',
    
    # Maharashtra
    'mumbai': 'maharashtra',
    'navi mumbai': 'maharashtra',
    'pune': 'maharashtra',
    'nagpur': 'maharashtra',
    'thane': 'maharashtra',
    
    # Himachal Pradesh
    'shimla': 'himachal pradesh',
    'manali': 'himachal pradesh',
    'dharamshala': 'himachal pradesh',
    'kullu': 'himachal pradesh',
    
    # Delhi NCR
    'delhi': 'delhi ncr',
    'new delhi': 'delhi ncr',
    'noida': 'delhi ncr',
    'gurgaon': 'delhi ncr',
    'gurugram': 'delhi ncr',
    'faridabad': 'delhi ncr',
    
    # Tamil Nadu
    'chennai': 'tamil nadu',
    'madras': 'tamil nadu',
    'coimbatore': 'tamil nadu',
    'madurai': 'tamil nadu',
    
    # West Bengal
    'kolkata': 'west bengal',
    'calcutta': 'west bengal',
    'darjeeling': 'west bengal',
    
    # Telangana
    'hyderabad': 'telangana',
    'secunderabad': 'telangana',
    
    # Kerala
    'kochi': 'kerala',
    'thiruvananthapuram': 'kerala',
    'kozhikode': 'kerala',
    'wayanad': 'kerala',
    
    # Uttarakhand
    'dehradun': 'uttarakhand',
    'haridwar': 'uttarakhand',
    'rishikesh': 'uttarakhand',
    'nainital': 'uttarakhand',
    
    # Assam
    'guwahati': 'assam',
    'dispur': 'assam',
    
    # Bihar
    'patna': 'bihar',
    'gaya': 'bihar',
    
    # Odisha
    'bhubaneswar': 'odisha',
    'cuttack': 'odisha',
    'puri': 'odisha',
    
    # Goa
    'panaji': 'goa',
    'margao': 'goa',
    'vasco': 'goa',
}

# ========================================================================
# VERIFICATION SCRAPER CLASS
# ========================================================================

class VerificationScraper:
    """
    Scraper for the Verification Phase of Flood/Cloudburst Fusion System.
    Collects human signals from news + Reddit to validate XGBoost predictions.
    ✨ Enhanced with parallel scraping and automatic fallback.
    """

    def __init__(self, location="India", cache_file="news_cache.pkl", 
                 cache_hours=3, hours_lookback=12, enable_fallback=True,
                 min_articles_threshold=5, max_workers=10):
        self.location = location
        self.original_location = location
        self.enable_fallback = enable_fallback
        self.min_articles_threshold = min_articles_threshold
        self.max_workers = max_workers
        self.fallback_used = False
        
        self.keywords = ['flood', 'cloudburst', 'flashflood', 'heavy rain',
                        'deluge', 'torrential rain', 'waterlogging', 'inundation',
                        'downpour', 'monsoon']
        self.cache_file = cache_file
        self.cache_hours = cache_hours
        self.hours_lookback = hours_lookback
        self.location_filters = self._get_location_filters(location)
        
        # Thread-safe lock for collecting results
        self.lock = threading.Lock()

    def _get_fallback_location(self, location):
        """Get state-level fallback for a city."""
        loc_lower = location.lower().strip()
        return LOCATION_HIERARCHY.get(loc_lower)

    def _get_location_filters(self, location):
        """Generate location-specific keywords for filtering."""
        location_map = {
            'bengaluru': ['bengaluru', 'bangalore', 'karnataka', 'bnglr', 'blr', 'bengalore', 'whitefield', 'koramangala'],
            'bangalore': ['bengaluru', 'bangalore', 'karnataka', 'bnglr', 'blr', 'bengalore'],
            'karnataka': ['karnataka', 'bengaluru', 'bangalore', 'mysore', 'mysuru', 'mangalore', 'mangaluru', 'hubli', 'belgaum', 'belgavi'],
            'himachal pradesh': ['himachal', 'shimla', 'shimmer', 'kullu', 'manali', 'dharamshala', 'dharamsala', 'hp', 'kangra', 'kinnaur', 'mandi'],
            'shimla': ['shimla', 'shimmer', 'himachal', 'hp', 'kullu', 'kinnuar'],
            'mumbai': ['mumbai', 'bombay', 'maharashtra', 'thane', 'navi mumbai', 'pune', 'mh'],
            'maharashtra': ['maharashtra', 'mumbai', 'pune', 'nagpur', 'thane', 'nashik', 'aurangabad', 'mh'],
            'navi mumbai': ['navi mumbai', 'new mumbai', 'panvel', 'thane', 'vashi', 'belapur', 'nerul', 'kharghar', 'kalamboli', 'raigad'],
            'delhi': ['delhi', 'new delhi', 'ncr', 'yamuna', 'noida', 'gurgaon', 'gurugram', 'faridabad', 'delhi ncr'],
            'delhi ncr': ['delhi', 'ncr', 'noida', 'gurgaon', 'gurugram', 'faridabad', 'ghaziabad', 'greater noida'],
            'chennai': ['chennai', 'madras', 'tamil nadu', 'tn', 'tamilnadu', 'chengalpattu'],
            'tamil nadu': ['tamil nadu', 'tn', 'chennai', 'coimbatore', 'madurai', 'salem', 'tiruchirappalli', 'tirunelveli'],
            'kolkata': ['kolkata', 'calcutta', 'west bengal', 'wb', 'darjeeling'],
            'west bengal': ['west bengal', 'wb', 'kolkata', 'calcutta', 'darjeeling', 'siliguri', 'asansol'],
            'hyderabad': ['hyderabad', 'telangana', 'secunderabad', 'cyberabad', 'ts'],
            'telangana': ['telangana', 'ts', 'hyderabad', 'warangal', 'nizamabad', 'karimnagar'],
            'kerala': ['kerala', 'kochi', 'cochin', 'thiruvananthapuram', 'kozhikode', 'wayanad', 'idukki', 'ernakulam', 'kottayam'],
            'uttarakhand': ['uttarakhand', 'uk', 'dehradun', 'haridwar', 'rishikesh', 'nainital', 'kedarnath', 'uttarkashi', 'almora'],
            'assam': ['assam', 'guwahati', 'brahmaputra', 'kaziranga', 'dispur', 'silchar'],
            'bihar': ['bihar', 'patna', 'ganga', 'kosi', 'muzaffarpur', 'darbhanga', 'bhagalpur'],
            'odisha': ['odisha', 'orissa', 'bhubaneswar', 'puri', 'cuttack', 'rourkela', 'balasore'],
            'goa': ['goa', 'panaji', 'margao', 'vasco', 'mapusa', 'ponda'],
            'india': []
        }

        loc_lower = location.lower().strip()
        for key, aliases in location_map.items():
            if key in loc_lower or loc_lower in key:
                if aliases:
                    print(f"🎯 Location: {location}")
                    print(f"📍 Filters: {', '.join(aliases)}\n")
                else:
                    print(f"🎯 Location: {location}")
                    print(f"📍 Filters: All India\n")
                return aliases

        return [location.lower()]

    def _is_location_relevant(self, text):
        """Check if text mentions the target location."""
        if not self.location_filters:
            return True
        if text is None:
            return False
        text_lower = text.lower()
        return any(loc_filter in text_lower for loc_filter in self.location_filters)

    def _is_recent(self, pub_date):
        """Check if article is within lookback window."""
        if not isinstance(pub_date, datetime):
            return True
        cutoff = datetime.now() - timedelta(hours=self.hours_lookback)
        return pub_date >= cutoff

    def _clean_text(self, text):
        """Clean text for processing."""
        if text is None:
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        return text

    def _load_cache(self):
        """Load cached data if valid."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cached_df = cache_data['data']
                    cache_time = cache_data['timestamp']
                    cached_location = cache_data.get('location', 'Unknown')

                    time_diff = datetime.now() - cache_time
                    
                    # Check if location matches
                    if cached_location.lower() != self.original_location.lower():
                        print(f"⚠️  Location mismatch detected!")
                        print(f"   Cache location: {cached_location}")
                        print(f"   Current location: {self.original_location}")
                        print(f"   Clearing cache for fresh data...\n")
                        os.remove(self.cache_file)
                        return None, None

                    if time_diff < timedelta(hours=self.cache_hours):
                        hours_left = self.cache_hours - time_diff.total_seconds()/3600
                        print(f"✓ Using cached data from {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"  Cache valid for {hours_left:.1f} more hours\n")
                        return cached_df, cache_time
                    else:
                        print(f"⚠️  Cache expired. Fetching fresh data...\n")
                        os.remove(self.cache_file)
            except Exception as e:
                print(f"⚠️  Cache error: {e}\n")
        return None, None

    def _save_cache(self, df):
        """Save scraped data to cache with location info."""
        try:
            cache_data = {
                'data': df, 
                'timestamp': datetime.now(),
                'location': self.original_location
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Data cached for {self.original_location} (valid for {self.cache_hours} hours)")
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")

    # ========================================================================
    # ✨ NEW: PARALLEL SCRAPING METHODS
    # ========================================================================

    def _scrape_single_keyword_news(self, keyword):
        """Scrape Google News for a single keyword (parallel-safe)."""
        news_list = []
        try:
            query = f'{keyword} {self.location}'
            query_encoded = query.replace(' ', '+')
            url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-IN&gl=IN&ceid=IN:en"

            feed = feedparser.parse(url)
            entries_found = 0

            for entry in feed.entries[:50]:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except:
                    pub_date = datetime.now()

                if not self._is_recent(pub_date):
                    continue

                title = entry.title
                summary = entry.get('summary', entry.title)
                source = entry.get('source', {}).get('title', 'Unknown')

                # Rain keyword filter
                combined_text = (title + ' ' + summary).lower()
                if not any(kw in combined_text for kw in self.keywords):
                    continue

                # Location relevance check
                if not (self._is_location_relevant(title) or self._is_location_relevant(summary)):
                    continue

                news_list.append({
                    'published_date': pub_date,
                    'title': title,
                    'summary': self._clean_text(summary),
                    'raw_text': summary,
                    'link': entry.link,
                    'source': source,
                    'keyword': keyword,
                    'data_source': 'Google News'
                })
                entries_found += 1

            return keyword, entries_found, news_list

        except Exception as e:
            return keyword, 0, []

    def scrape_google_news(self):
        """Scrape Google News RSS feeds with PARALLEL processing."""
        print("📰 Scraping Google News RSS (Parallel)...\n")
        print(f"Keywords being searched: {', '.join(self.keywords)}")
        print(f"Workers: {self.max_workers}\n")

        all_news = []
        
        # ✨ Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_keyword = {
                executor.submit(self._scrape_single_keyword_news, kw): kw 
                for kw in self.keywords
            }
            
            for future in as_completed(future_to_keyword):
                keyword, count, news_list = future.result()
                if count > 0:
                    print(f"  ✓ '{keyword}': {count} articles found")
                    all_news.extend(news_list)
                time.sleep(0.1)  # Small delay between completions

        df = pd.DataFrame(all_news)
        if not df.empty:
            df = df.drop_duplicates(subset=['title'], keep='first')
            df = df.sort_values('published_date', ascending=False)
            print(f"\n✓ Total from Google News: {len(df)} articles\n")
        else:
            print(f"\n⚠️  No articles found from Google News\n")

        return df

    def _scrape_single_subreddit(self, subreddit):
        """Scrape a single subreddit (parallel-safe)."""
        posts_list = []
        try:
            url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=100"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                posts = data['data']['children']
                entries_found = 0

                for post in posts:
                    post_data = post['data']
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    combined = (title + ' ' + selftext).lower()

                    # Filter by keywords
                    if not any(kw in combined for kw in self.keywords):
                        continue

                    # Check recency
                    created_utc = datetime.fromtimestamp(post_data['created_utc'])
                    if not self._is_recent(created_utc):
                        continue

                    # Location relevance check
                    if not (self._is_location_relevant(title) or self._is_location_relevant(selftext)):
                        continue

                    posts_list.append({
                        'published_date': created_utc,
                        'title': title,
                        'summary': self._clean_text(selftext[:500]),
                        'raw_text': selftext[:500],
                        'link': f"https://reddit.com{post_data['permalink']}",
                        'source': f"r/{subreddit}",
                        'keyword': 'reddit',
                        'data_source': 'Reddit'
                    })
                    entries_found += 1

                return subreddit, entries_found, posts_list

        except Exception as e:
            pass

        return subreddit, 0, []

    def scrape_reddit(self):
        """Scrape Reddit posts with PARALLEL processing."""
        print("🔴 Scraping Reddit (Parallel)...\n")

        subreddits = ['india', 'IndiaSpeaks', 'bangalore', 'mumbai', 'delhi',
                     'hyderabad', 'chennai', 'kolkata', 'Kerala', 'UttarPradesh',
                     'Bihar', 'Odisha', 'Assam', 'weather', 'IndianWeather']

        print(f"Workers: {self.max_workers}\n")
        
        all_posts = []
        
        # ✨ Parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sub = {
                executor.submit(self._scrape_single_subreddit, sub): sub 
                for sub in subreddits
            }
            
            for future in as_completed(future_to_sub):
                subreddit, count, posts_list = future.result()
                if count > 0:
                    print(f"  ✓ r/{subreddit}: {count} posts found")
                    all_posts.extend(posts_list)
                time.sleep(0.1)

        df = pd.DataFrame(all_posts)
        if not df.empty:
            df = df.drop_duplicates(subset=['title'], keep='first')
            df = df.sort_values('published_date', ascending=False)
            print(f"\n✓ Total from Reddit: {len(df)} posts\n")
        else:
            print(f"\n⚠️  No posts found from Reddit\n")

        return df

    # ========================================================================
    # ✨ NEW: AUTOMATIC FALLBACK LOGIC
    # ========================================================================

    def _attempt_scrape_with_location(self, location):
        """Attempt to scrape with a given location."""
        self.location = location
        self.location_filters = self._get_location_filters(location)
        
        google_news = self.scrape_google_news()
        reddit_posts = self.scrape_reddit()
        
        all_data = pd.concat([google_news, reddit_posts], ignore_index=True)
        
        if not all_data.empty:
            all_data = all_data.drop_duplicates(subset=['title'], keep='first')
            all_data = all_data.sort_values('published_date', ascending=False)
        
        return all_data

    def scrape_all(self, force_refresh=False):
        """Main scraping function with automatic fallback."""
        print("=" * 70)
        print("VERIFICATION PHASE - Human Signal Collection")
        print("=" * 70)
        print(f"Location: {self.original_location}")
        print(f"Lookback Window: Last {self.hours_lookback} hours")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Parallel Workers: {self.max_workers}")
        print(f"Fallback Enabled: {self.enable_fallback}")
        print("=" * 70 + "\n")

        # Check cache
        if not force_refresh:
            cached_df, cache_time = self._load_cache()
            if cached_df is not None and len(cached_df) > 0:
                return cached_df

        # Scrape fresh data
        print("🔄 Fetching fresh data...\n")
        
        # ✨ Try primary location first
        all_data = self._attempt_scrape_with_location(self.original_location)
        
        # ✨ Check if fallback needed
        if self.enable_fallback and len(all_data) < self.min_articles_threshold:
            fallback_location = self._get_fallback_location(self.original_location)
            
            if fallback_location:
                print("\n" + "=" * 70)
                print(f"⚠️  INSUFFICIENT DATA ({len(all_data)} articles < {self.min_articles_threshold} threshold)")
                print(f"🔄 Attempting STATE-LEVEL FALLBACK: {fallback_location.upper()}")
                print("=" * 70 + "\n")
                
                fallback_data = self._attempt_scrape_with_location(fallback_location)
                
                if len(fallback_data) > len(all_data):
                    print(f"\n✅ FALLBACK SUCCESSFUL: {len(fallback_data)} articles found\n")
                    all_data = fallback_data
                    self.fallback_used = True
                else:
                    print(f"\n⚠️  Fallback yielded similar results. Using original.\n")

        if not all_data.empty:
            self._save_cache(all_data)

            print("\n" + "=" * 70)
            print(f"✅ SCRAPING COMPLETE - {len(all_data)} articles/posts collected")
            if self.fallback_used:
                print(f"   ℹ️  Data includes state-level fallback results")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("⚠️  NO DATA FOUND")
            print("=" * 70)
            print("\nPossible Reasons:")
            print(f"1. No flooding reported in {self.original_location} in last {self.hours_lookback}h")
            print("2. Try increasing hours_lookback (e.g., hours_lookback=24)")
            print("3. Try broader location (e.g., 'Karnataka' instead of 'Bengaluru')")
            print("4. Check internet connection")
            print("\n💡 This is EXPECTED for locations without active flooding!")

        return all_data


# ========================================================================
# USAGE EXAMPLE
# ========================================================================

if __name__ == "__main__":
    print("🌊 FLOOD/CLOUDBURST FUSION SYSTEM - VERIFICATION PHASE\n")
    print("✨ Enhanced with Parallel Scraping & Auto Fallback\n")

    # ⚙️ CONFIGURE THIS
    location = "delhi"  # 🔥 Change to your target location
    
    # Clear cache for this location
    clear_cache_for_new_location()
    
    # Create scraper with enhanced features
    scraper = VerificationScraper(
        location=location,
        cache_hours=3,
        hours_lookback=24,
        enable_fallback=True,  # ✨ Enable automatic fallback
        min_articles_threshold=5,  # ✨ Minimum articles before fallback
        max_workers=10  # ✨ Parallel workers (adjust based on your system)
    )

    # Scrape data
    verification_data = scraper.scrape_all(force_refresh=False)

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    if not verification_data.empty:
        print("\n" + "=" * 70)
        print("📊 DATA STATISTICS")
        print("=" * 70 + "\n")

        print(verification_data.head(10))

        print(f"\n\nTotal Items: {len(verification_data)}")
        print(f"\nBy Source:")
        print(verification_data['data_source'].value_counts())
        print(f"\nTime Range:")
        print(f"  Oldest: {verification_data['published_date'].min()}")
        print(f"  Newest: {verification_data['published_date'].max()}")

        # Prepare for DistilBERT
        print("\n" + "=" * 70)
        print("🤖 PREPARING FOR DISTILBERT TEXT ANALYSIS")
        print("=" * 70 + "\n")

        verification_data['text_for_model'] = (
            verification_data['title'] + ' ' + verification_data['summary']
        )

        text_samples = verification_data['text_for_model'].tolist()

        print(f"✓ {len(text_samples)} text samples ready")
        print("\n📝 Sample Texts:")
        for i, text in enumerate(text_samples[:3], 1):
            print(f"\n{i}. {text[:150]}...")

        # Save
        verification_data.to_csv('verification_data.csv', index=False)
        print(f"\n💾 Saved to 'verification_data.csv'\n")

        print("=" * 70)
        print("✅ NEXT: Feed to DistilBERT → Calculate text_signal_score")
        print("=" * 70)

    else:
        print("No data to display.")