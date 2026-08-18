#!/usr/bin/env python3
"""Fast training for hit_model using sample"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "C:/Users/Hugo.DESKTOP-QQG83V5/Downloads/plydb-fun-baseball-analyst-main/data/pybaseball/statcast/statcast_multiseason_batter_game_level.parquet"
MODEL_PATH = "C:/Users/Hugo.DESKTOP-QQG83V5/Downloads/plydb-fun-baseball-analyst-main/mlb_analyst/hit_model.joblib"

EXCLUDE_COLS = ['batter', 'pitcher', 'game_pk', 'game_date', 'season', 'home_team', 'away_team', 'hit_in_game', 'pa_count_this_game']

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} rows")

# Sample for speed - use 50k rows
df = df.sample(n=50000, random_state=42)
print(f"Using sample: {len(df)} rows")

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X = df[feature_cols].copy()
y = df['hit_in_game'].astype(int).copy()

# Encode categorical
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Fill NaN
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Simple split
train_mask = df['season'].isin([2023, 2024])
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

print("Training GradientBoosting...")
m = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
m.fit(X_train, y_train)

print("Calibrating...")
cal = CalibratedClassifierCV(m, method='isotonic', cv=2)
cal.fit(X_train, y_train)

from sklearn.metrics import log_loss
proba = cal.predict_proba(X_test)[:,1]
print(f"LogLoss: {log_loss(y_test, proba):.4f}")

joblib.dump({'model': cal, 'features': list(X.columns)}, MODEL_PATH)
print(f"Saved to {MODEL_PATH}")