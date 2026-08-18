#!/usr/bin/env python3
"""
Train hit_model.joblib - Batter hit probability model

Uses statcast_multiseason_batter_game_level.parquet with target 'hit_in_game'
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, using GradientBoostingClassifier")

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "pybaseball" / "statcast" / "statcast_multiseason_batter_game_level.parquet"
MODEL_PATH = Path(__file__).parent / "hit_model.joblib"

# Feature columns (exclude target and identifiers)
EXCLUDE_COLS = ['batter', 'pitcher', 'game_pk', 'game_date', 'season', 'home_team', 'away_team', 'hit_in_game', 'pa_count_this_game']

def load_and_prepare_data():
    """Load data and prepare features/target"""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Hit rate: {df['hit_in_game'].mean():.3f}")
    
    # Features
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df['hit_in_game'].astype(int).copy()
    
    # Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle NaN in rolling features (early season) - only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Time-based split: 2023-2024 train, 2025 test
    train_mask = df['season'].isin([2023, 2024])
    test_mask = df['season'] == 2025
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTrain: {len(X_train):,} samples (seasons 2023-2024)")
    print(f"Test:  {len(X_test):,} samples (season 2025)")
    print(f"Features: {list(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate models"""
    
    models = {}
    
    if XGB_AVAILABLE:
        models['xgb'] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
    
    models['gbm'] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    models['rf'] = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    results = {}
    best_model = None
    best_score = float('inf')
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  LogLoss:  {ll:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")
        print(f"  Brier:    {brier:.4f}")
        
        results[name] = {'model': model, 'accuracy': acc, 'logloss': ll, 'auc': auc, 'brier': brier}
        
        if ll < best_score:
            best_score = ll
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (logloss={best_score:.4f})")
    
    # Calibrate best model
    print("\nCalibrating best model with isotonic regression...")
    calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Evaluate calibrated
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    ll_cal = log_loss(y_test, y_proba_cal)
    brier_cal = brier_score_loss(y_test, y_proba_cal)
    print(f"Calibrated LogLoss: {ll_cal:.4f}, Brier: {brier_cal:.4f}")
    
    return calibrated, feature_cols, results

def save_model(model, feature_cols, results):
    """Save model with metadata"""
    model_data = {
        'model': model,
        'features': feature_cols,
        'training_results': results,
        'model_type': 'hit_probability',
        'target': 'hit_in_game'
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING HIT MODEL")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
    model, feature_cols, results = train_model(X_train, y_train, X_test, y_test)
    save_model(model, feature_cols, results)
    
    print("\nDone!")