#!/usr/bin/env python3
"""Train win_model - much faster using game-level data"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

PA_PATH = "C:/Users/Hugo.DESKTOP-QQG83V5/Downloads/plydb-fun-baseball-analyst-main/data/pybaseball/statcast/statcast_multiseason_pa_level_model_ready.parquet"
MODEL_PATH = "C:/Users/Hugo.DESKTOP-QQG83V5/Downloads/plydb-fun-baseball-analyst-main/mlb_analyst/win_model.joblib"

print("Loading PA data...")
df = pd.read_parquet(PA_PATH)
print(f"Loaded {len(df):,} PA rows")

# Build game-level dataset
print("Building game-level features...")
games = df.groupby('game_pk').apply(lambda g: pd.Series({
    'game_date': g['game_date'].iloc[0],
    'season': g['season'].iloc[0],
    'home_team': g['home_team'].iloc[0],
    'away_team': g['away_team'].iloc[0],
    'home_win': (g['post_home_score'].max() > g['post_away_score'].max()).astype(int),
    # Home batting (bottom of inning)
    'home_batting_hit_rate': g[g['inning_topbot']=='Bot']['is_hit'].mean(),
    'home_batting_xba': g[g['inning_topbot']=='Bot']['estimated_ba_using_speedangle'].mean(),
    'home_batting_k_rate': g[g['inning_topbot']=='Bot']['is_strikeout'].mean(),
    'home_batting_bb_rate': g[g['inning_topbot']=='Bot']['is_walk'].mean(),
    'home_batting_hardhit_rate': g[g['inning_topbot']=='Bot']['is_hard_hit'].mean(),
    # Away batting (top of inning)
    'away_batting_hit_rate': g[g['inning_topbot']=='Top']['is_hit'].mean(),
    'away_batting_xba': g[g['inning_topbot']=='Top']['estimated_ba_using_speedangle'].mean(),
    'away_batting_k_rate': g[g['inning_topbot']=='Top']['is_strikeout'].mean(),
    'away_batting_bb_rate': g[g['inning_topbot']=='Top']['is_walk'].mean(),
    'away_batting_hardhit_rate': g[g['inning_topbot']=='Top']['is_hard_hit'].mean(),
    # Home pitching (when away bats)
    'home_pitcher_k_rate': g[g['inning_topbot']=='Top']['is_strikeout'].mean(),
    'home_pitcher_bb_rate': g[g['inning_topbot']=='Top']['is_walk'].mean(),
    'home_pitcher_xba_allowed': g[g['inning_topbot']=='Top']['estimated_ba_using_speedangle'].mean(),
    'home_pitcher_hardhit_allowed': g[g['inning_topbot']=='Top']['is_hard_hit'].mean(),
    'home_pitcher_velo': g[g['inning_topbot']=='Top']['avg_release_speed'].mean(),
    'home_pitcher_spin': g[g['inning_topbot']=='Top']['avg_spin_rate'].mean(),
    # Away pitching (when home bats)
    'away_pitcher_k_rate': g[g['inning_topbot']=='Bot']['is_strikeout'].mean(),
    'away_pitcher_bb_rate': g[g['inning_topbot']=='Bot']['is_walk'].mean(),
    'away_pitcher_xba_allowed': g[g['inning_topbot']=='Bot']['estimated_ba_using_speedangle'].mean(),
    'away_pitcher_hardhit_allowed': g[g['inning_topbot']=='Bot']['is_hard_hit'].mean(),
    'away_pitcher_velo': g[g['inning_topbot']=='Bot']['avg_release_speed'].mean(),
    'away_pitcher_spin': g[g['inning_topbot']=='Bot']['avg_spin_rate'].mean(),
})).reset_index()

print(f"Games: {len(games)}")

# Add park factors
games['park_hit_factor'] = 1.0
games['park_hr_factor'] = 1.0

# Rolling features per team
feature_cols = [c for c in games.columns if c not in ['game_pk','game_date','season','home_team','away_team','home_win']]
for prefix, team_col in [('home_','home_team'), ('away_','away_team')]:
    team_feats = [c for c in feature_cols if c.startswith(prefix)]
    for f in team_feats:
        games[f'{f}_roll'] = games.groupby(team_col)[f].transform(lambda x: x.expanding().mean().shift(1))

# Fill NaN
roll_cols = [c for c in games.columns if c.endswith('_roll')]
games[roll_cols] = games[roll_cols].fillna(games[roll_cols].mean())

# Final features
final_feats = [c for c in games.columns if c.endswith('_roll')] + ['park_hit_factor','park_hr_factor']
print(f"Features: {len(final_feats)}")

# Time split
train = games['season'].isin([2023, 2024])
test = games['season'] == 2025

X_train = games[train][final_feats]
y_train = games[train]['home_win']
X_test = games[test][final_feats]
y_test = games[test]['home_win']

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train win rate: {y_train.mean():.3f}")

# Train
print("Training...")
m = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
m.fit(X_train, y_train)

print("Calibrating...")
cal = CalibratedClassifierCV(m, method='isotonic', cv=2)
cal.fit(X_train, y_train)

ll = log_loss(y_test, cal.predict_proba(X_test)[:,1])
print(f"LogLoss: {ll:.4f}")

joblib.dump({'model': cal, 'features': final_feats}, MODEL_PATH)
print(f"Saved to {MODEL_PATH}")