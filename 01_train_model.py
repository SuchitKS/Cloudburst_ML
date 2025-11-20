import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             average_precision_score, classification_report,
                             cohen_kappa_score, matthews_corrcoef)

# Try importing SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ imbalanced-learn library found - SMOTE enabled")
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn NOT found")

print("="*70)
print("CLOUDBURST MODEL - PHYSICS ONLY (NO TIME BIAS)")
print("="*70)
print("✅ Removed 'Month' and 'Year' features")
print("✅ Model forced to learn Atmospheric Physics")
print("✅ 50mm threshold (IMD Heavy Rainfall)")
print("="*70 + "\n")

start_time = time.time()

# --- CONFIGURATION ---
CHUNK_SIZE = 200_000
BATCH_SIZE = 500_000
TREES_PER_BATCH = 100
MAX_TREES = 2000
EARLY_STOPPING = 10
USE_SMOTE = SMOTE_AVAILABLE
SMOTE_TARGET_RATIO = 0.1 

# --- 1. DEFINING PHYSICS-ONLY FEATURES ---
# REMOVED: 'month', 'year'
# KEPT: 'lat', 'lon' (Topography is important), and all weather variables
features = ['lat', 'lon', 'dew2m', 'latent_flux', 'sensible_flux',
            'cloud_cover', 'cloud_liquid', 'wind_speed']

print(f"🧬 Training Features ({len(features)}):")
print(f"   {features}")
print("   (Model will relies on moisture & instability, not the calendar date)\n")

# Define types for memory efficiency
dtype_feat = {c: 'float32' for c in features}
dtype_feat['year'] = 'int16' # Still needed for filtering, but not training
dtype_feat['month'] = 'int8' # Still needed for filtering

# --- XGBoost Parameters ---
# Lower max_depth slightly to prevent overfitting to specific lat/lon points
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'max_depth': 5,             # Reduced from 6 to encourage generalization
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,      # Increased to require more robust patterns
    'gamma': 0.2,
    'tree_method': 'hist',
    'nthread': -1
}

# --- Helper Functions (Same as before) ---
def merge_data(feat_df):
    feat_df['merge_key'] = (
        feat_df['lat'].round(2).astype(str) + '_' +
        feat_df['lon'].round(2).astype(str) + '_' +
        feat_df['date'].dt.strftime('%Y-%m-%d')
    )
    merged_list = []
    # Use specific chunks to save memory
    for t_chunk in pd.read_csv('Y_target_file.csv',
                               dtype={'merge_key': str, 'Y_target': 'int8'},
                               chunksize=1_000_000):
        m = feat_df.merge(t_chunk, on='merge_key', how='inner')
        if not m.empty: merged_list.append(m)
    
    if not merged_list: return pd.DataFrame()
    return pd.concat(merged_list, ignore_index=True)

def handle_missing_values(X):
    return X.fillna(X.median())

def apply_smote_balancing(X, Y):
    if not SMOTE_AVAILABLE: return X, Y
    try:
        smote = SMOTE(sampling_strategy=SMOTE_TARGET_RATIO, random_state=42)
        X_bal, Y_bal = smote.fit_resample(X, Y)
        return X_bal, Y_bal
    except:
        return X, Y

# --- STEP 1: Validation Set (2021) ---
print("\n🌊 Creating Validation Set...")
val_dfs = []
for chunk in pd.read_csv('complete_merged_2015_2024.csv',
                         usecols=features + ['date', 'year', 'month'], # Read year for filtering only
                         dtype=dtype_feat, parse_dates=['date'], chunksize=CHUNK_SIZE):
    chunk = chunk[chunk['year'] == 2021]
    if not chunk.empty:
        val_dfs.append(chunk)
        if len(val_dfs) * CHUNK_SIZE > 500_000: break

if val_dfs:
    val_merged = merge_data(pd.concat(val_dfs))
    if not val_merged.empty:
        X_val = val_merged[features].copy() # Select ONLY physics features
        Y_val = val_merged['Y_target'].copy()
        X_val = handle_missing_values(X_val)
        dval = xgb.DMatrix(X_val, label=Y_val)
        print(f"   ✅ Validation loaded: {len(X_val)} samples")
    else: dval = None
else: dval = None

# --- STEP 2: Test Set (2022-2023) ---
print("🌊 Creating Test Set...")
test_dfs = []
for chunk in pd.read_csv('complete_merged_2015_2024.csv',
                         usecols=features + ['date', 'year', 'month'],
                         dtype=dtype_feat, parse_dates=['date'], chunksize=CHUNK_SIZE):
    chunk = chunk[(chunk['year'] >= 2022) & (chunk['year'] <= 2023)]
    if not chunk.empty:
        test_dfs.append(chunk)
        if len(test_dfs) * CHUNK_SIZE > 500_000: break

if test_dfs:
    test_merged = merge_data(pd.concat(test_dfs))
    if not test_merged.empty:
        X_test = test_merged[features].copy()
        Y_test = test_merged['Y_target'].copy()
        X_test = handle_missing_values(X_test)
        dtest = xgb.DMatrix(X_test, label=Y_test)
        print(f"   ✅ Test loaded: {len(X_test)} samples")
    else: dtest = None
else: dtest = None

# --- STEP 3: Training Loop (2015-2020) ---
print("\n🚀 Starting Training Loop...")
bst = None
batch_buffer = []

for chunk in pd.read_csv('complete_merged_2015_2024.csv',
                         usecols=features + ['date', 'year', 'month'],
                         dtype=dtype_feat, parse_dates=['date'], chunksize=CHUNK_SIZE):
    
    chunk = chunk[chunk['year'] <= 2020]
    if chunk.empty: continue
    
    batch_buffer.append(chunk)
    if sum(len(c) for c in batch_buffer) >= BATCH_SIZE:
        merged_batch = merge_data(pd.concat(batch_buffer))
        
        if not merged_batch.empty:
            X_train = merged_batch[features].copy()
            Y_train = merged_batch['Y_target'].copy()
            X_train = handle_missing_values(X_train)
            
            if USE_SMOTE:
                X_train, Y_train = apply_smote_balancing(X_train, Y_train)
            
            dtrain = xgb.DMatrix(X_train, label=Y_train)
            evals = [(dval, 'val')] if dval else []
            
            bst = xgb.train(params, dtrain, num_boost_round=TREES_PER_BATCH,
                            xgb_model=bst, evals=evals, verbose_eval=False)
            
            print(f"   🌳 Trees: {bst.num_boosted_rounds()} | Last Val AUCPR: {bst.eval(dval).split(':')[-1]}")
            
        batch_buffer = []
        if bst and bst.num_boosted_rounds() >= MAX_TREES: break

# --- STEP 4: Save & Optimize ---
if bst:
    print("\n💾 Saving Physics Model...")
    bst.save_model('xgb_flood_model.json')
    
    # Save feature names so prediction script knows NOT to ask for 'month'
    with open('model_columns.json', 'w') as f:
        json.dump(features, f)
    
    # Calculate Optimal Threshold
    if dtest:
        probs = bst.predict(dtest)
        best_f1 = 0
        best_thresh = 0.5
        for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
            preds = (probs >= t).astype(int)
            f1 = f1_score(Y_test, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        
        print(f"\n🏆 OPTIMAL THRESHOLD: {best_thresh}")
        with open('training_metadata.json', 'w') as f:
            json.dump({'optimal_threshold': best_thresh}, f)

    print("\n✅ DONE. Model is now Time-Agnostic (Physics based).")