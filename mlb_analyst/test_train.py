#!/usr/bin/env python3
"""Quick test of data loading and prep"""

import pandas as pd
import numpy as np

df = pd.read_parquet('data/pybaseball/statcast/statcast_multiseason_batter_game_level.parquet')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('Hit rate:', df['hit_in_game'].mean())

EXCLUDE_COLS = ['batter', 'pitcher', 'game_pk', 'game_date', 'season', 'home_team', 'away_team', 'hit_in_game', 'pa_count_this_game']
X = df.drop(columns=EXCLUDE_COLS)
numeric_cols = X.select_dtypes(include=[np.number]).columns
print('Numeric cols:', len(numeric_cols))
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
print('Filled NaN')

train_mask = df['season'].isin([2023, 2024])
test_mask = df['season'] == 2025

X_train = X[train_mask]
y_train = df[train_mask]['hit_in_game']
X_test = X[test_mask]
y_test = df[test_mask]['hit_in_game']

print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print(f'Train hit rate: {y_train.mean():.3f}')
print(f'Test hit rate: {y_test.mean():.3f}')
print('Done!')