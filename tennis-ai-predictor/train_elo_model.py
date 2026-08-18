#!/usr/bin/env python3
"""
Train a surface-specific Elo model for tennis match prediction.

Reads Jeff Sackmann Match Charting Project data, computes Elo ratings per player
per surface over time, and trains a classifier to predict match winners.

Output: tennis-ai-predictor/models/tennis_elo_model.pkl (compatible with app.py)
"""

import os
import sys
import pickle
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# ─── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
MATCHES_M = r'C:\Users\Hugo.DESKTOP-QQG83V5\Downloads\tennis_MatchChartingProject-master\tennis_MatchChartingProject-master\charting-m-matches.csv'
MATCHES_W = r'C:\Users\Hugo.DESKTOP-QQG83V5\Downloads\tennis_MatchChartingProject-master\tennis_MatchChartingProject-master\charting-w-matches.csv'
MODEL_PATH = os.path.join(MODEL_DIR, 'tennis_elo_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Elo Parameters ─────────────────────────────────────────────────────
INITIAL_ELO = 1500
K_FACTOR = 32
SURFACE_K = {'Hard': 32, 'Clay': 28, 'Grass': 24, 'Carpet': 20}
HOME_ADVANTAGE = 0  # Neutral courts in charting data
MIN_MATCHES_FOR_ELO = 5  # Minimum matches before Elo is "reliable"

# ─── Helpers ────────────────────────────────────────────────────────────

def parse_date(date_str):
    """Parse YYYYMMDD date string."""
    try:
        return datetime.strptime(str(date_str).strip(), '%Y%m%d')
    except Exception:
        return None


def expected_score(elo_a, elo_b):
    """Expected score for player A vs B."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(elo_a, elo_b, score_a, k_factor):
    """Update Elo ratings after a match. score_a = 1 if A won, 0 if B won."""
    exp_a = expected_score(elo_a, elo_b)
    new_a = elo_a + k_factor * (score_a - exp_a)
    new_b = elo_b + k_factor * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def load_matches(csv_path, gender='M'):
    """Load and clean match data."""
    print(f"Loading {gender} matches from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Clean columns
    df.columns = df.columns.str.strip()
    
    # Parse date
    df['date'] = df['Date'].apply(parse_date)
    df = df.dropna(subset=['date'])
    
    # Standardize surface names
    surface_map = {
        'Hard': 'Hard', 'hard': 'Hard', 'Hard Court': 'Hard',
        'Clay': 'Clay', 'clay': 'Clay', 'Red Clay': 'Clay',
        'Grass': 'Grass', 'grass': 'Grass',
        'Carpet': 'Carpet', 'carpet': 'Carpet', 'Indoor': 'Carpet'
    }
    df['surface'] = df['Surface'].map(surface_map).fillna('Hard')
    
    # Winner: Player 1 is column 'Player 1', Player 2 is 'Player 2'
    # We need to determine winner from the data - check if there's a winner column
    # In charting data, typically the match winner isn't directly in matches.csv
    # We'll infer from the match_id or assume Player 1 is winner for now
    # Actually, let's check the match_id format - it often contains the winner
    
    # For now, we need to determine winner. The charting data doesn't have explicit winner in matches.csv
    # But we can infer from the point-by-point data or stats
    # Simpler approach: we'll use the Overview stats to see who won more points/sets
    # Actually, let's check if there's a "Winner" column or similar
    
    df['player1'] = df['Player 1'].str.strip()
    df['player2'] = df['Player 2'].str.strip()
    df['gender'] = gender
    
    # Keep only needed columns
    df = df[['match_id', 'date', 'player1', 'player2', 'surface', 'gender', 'Tournament', 'Round', 'Best of']].copy()
    df['best_of'] = pd.to_numeric(df['Best of'], errors='coerce').fillna(3).astype(int)
    
    return df.sort_values('date').reset_index(drop=True)


def determine_winner_from_stats(match_id, data_root):
    """Determine match winner from Overview stats (who won more sets/points)."""
    overview_path = os.path.join(data_root, 'charting-m-stats-Overview.csv')
    if not os.path.exists(overview_path):
        overview_path = os.path.join(data_root, 'charting-w-stats-Overview.csv')
    
    # This is expensive to do for every match - let's load once
    return None


def load_all_matches():
    """Load both men's and women's matches."""
    data_root = os.path.join(os.path.dirname(MATCHES_M))
    
    men = load_matches(MATCHES_M, 'M')
    women = load_matches(MATCHES_W, 'W')
    
    all_matches = pd.concat([men, women], ignore_index=True)
    all_matches = all_matches.sort_values('date').reset_index(drop=True)
    
    print(f"Total matches: {len(all_matches)} (M: {len(men)}, W: {len(women)})")
    print(f"Date range: {all_matches['date'].min()} to {all_matches['date'].max()}")
    print(f"Surfaces: {all_matches['surface'].value_counts().to_dict()}")
    
    return all_matches, data_root


def compute_elo_ratings(matches, data_root):
    """
    Compute surface-specific Elo ratings for each player over time.
    Returns a DataFrame with match_id, player1_elo, player2_elo, winner (1 or 0).
    """
    # Player -> Surface -> Elo rating
    player_elo = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))
    # Player -> Surface -> match count
    player_matches = defaultdict(lambda: defaultdict(int))
    
    # We need to determine winners. Let's load the Overview stats once.
    print("Loading Overview stats to determine winners...")
    overview_m = os.path.join(data_root, 'charting-m-stats-Overview.csv')
    overview_w = os.path.join(data_root, 'charting-w-stats-Overview.csv')
    
    overview_dfs = []
    if os.path.exists(overview_m):
        overview_dfs.append(pd.read_csv(overview_m))
    if os.path.exists(overview_w):
        overview_dfs.append(pd.read_csv(overview_w))
    
    if overview_dfs:
        overview = pd.concat(overview_dfs, ignore_index=True)
        # Aggregate by match_id and player to get total sets won
        # The 'set' column has values like '1', '2', '3', 'Total'
        # We'll use the 'Total' row for each player
        overview_total = overview[overview['set'] == 'Total'].copy()
        # Group by match_id, player to see who has more games/sets won
        # Actually, let's use the match-level data from matches.csv - 
        # the winner isn't directly there. We need another approach.
        
        # Alternative: load match stats (sets won) from charting-m-stats-Overview
        # The 'set' column values: '1', '2', '3', 'Total' - we need sets won
        # Let's check the data structure more carefully
        print(f"Overview shape: {overview.shape}")
        print(f"Unique sets: {overview['set'].unique()[:10]}")
    else:
        overview = None
    
    # Since determining winner from charting data is complex, let's use a simpler approach:
    # The match_id often encodes tournament/round. For training, we can use the fact that
    # higher-ranked/seeded player is often Player 1. But that's biased.
    # Better: Let's look at the point-by-point data for a sample to infer winner
    
    # Actually, for Elo training we need winners. Let's check if there's a simpler source.
    # The ATP/WTA match results are in the match charting project's "matches" files
    # but winner isn't explicit. Let me check the match_id format.
    
    # match_id format: YYYYMMDD-Gender-Tournament-Round-Player1-Player2
    # No winner info there.
    
    # We'll need to compute winner from the point-by-point data or stats.
    # Let's load a sample of point data to see if we can determine winner.
    
    print("Computing Elo ratings (assuming Player 1 wins for now - PLACEHOLDER)...")
    
    # PLACEHOLDER: We'll assume we can determine winner somehow
    # For now, let's create synthetic training data using Elo only
    # and later replace with actual winners
    
    results = []
    
    for idx, row in matches.iterrows():
        p1 = row['player1']
        p2 = row['player2']
        surface = row['surface']
        date = row['date']
        match_id = row['match_id']
        
        elo1 = player_elo[p1][surface]
        elo2 = player_elo[p2][surface]
        n1 = player_matches[p1][surface]
        n2 = player_matches[p2][surface]
        
        # Use surface-specific K-factor
        k = SURFACE_K.get(surface, K_FACTOR)
        
        # For training, we need the actual winner
        # Since we don't have it easily, we'll skip matches where we can't determine
        # For now, create a placeholder - we'll need to fix this
        
        results.append({
            'match_id': match_id,
            'date': date,
            'player1': p1,
            'player2': p2,
            'surface': surface,
            'elo1': elo1,
            'elo2': elo2,
            'n1': n1,
            'n2': n2,
            'k_factor': k,
        })
        
        # Update match counts
        player_matches[p1][surface] += 1
        player_matches[p2][surface] += 1
    
    return pd.DataFrame(results)


def determine_winners_from_points(data_root, match_ids):
    """
    Determine match winners by loading point-by-point data.
    This is slow but accurate.
    """
    print("Loading point data to determine winners...")
    
    # Load men's points
    points_files = [
        os.path.join(data_root, 'charting-m-points-2020s.csv'),
        os.path.join(data_root, 'charting-m-points-2010s.csv'),
        os.path.join(data_root, 'charting-m-points-to-2009.csv'),
        os.path.join(data_root, 'charting-w-points-2020s.csv'),
        os.path.join(data_root, 'charting-w-points-2010s.csv'),
        os.path.join(data_root, 'charting-w-points-to-2009.csv'),
    ]
    
    winners = {}
    
    for pf in points_files:
        if not os.path.exists(pf):
            continue
        print(f"  Processing {os.path.basename(pf)}...")
        # Read in chunks to handle large files
        for chunk in pd.read_csv(pf, chunksize=100000, usecols=['match_id', 'PtWinner']):
            # Last point winner for each match = match winner
            last_points = chunk.groupby('match_id').last()['PtWinner']
            for mid, winner in last_points.items():
                if mid in match_ids:
                    winners[mid] = winner  # 1 or 2
    
    print(f"Determined winners for {len(winners)} matches")
    return winners


def build_training_data(matches, data_root):
    """Build training dataset with Elo features and actual winners."""
    match_ids = set(matches['match_id'].tolist())
    
    # Get winners from point data
    winners = determine_winners_from_points(data_root, match_ids)
    
    # Now compute Elo with actual winners
    player_elo = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))
    player_matches = defaultdict(lambda: defaultdict(int))
    player_form = defaultdict(lambda: defaultdict(list))  # recent results for form
    
    training_rows = []
    
    for idx, row in matches.iterrows():
        match_id = row['match_id']
        if match_id not in winners:
            continue
            
        p1 = row['player1']
        p2 = row['player2']
        surface = row['surface']
        date = row['date']
        
        elo1 = player_elo[p1][surface]
        elo2 = player_elo[p2][surface]
        n1 = player_matches[p1][surface]
        n2 = player_matches[p2][surface]
        
        # Form: win rate in last 10 matches on this surface
        form1 = np.mean(player_form[p1][surface][-10:]) if player_form[p1][surface] else 0.5
        form2 = np.mean(player_form[p2][surface][-10:]) if player_form[p2][surface] else 0.5
        form_diff = form1 - form2
        
        # Rest days: days since last match for each player
        # We'd need to track last match date per player - skip for now
        rest_diff = 0.0
        
        # Winner: 1 if player1 won, 0 if player2 won
        winner = 1 if winners[match_id] == 1 else 0
        
        # Only use matches where both players have some history
        if n1 >= MIN_MATCHES_FOR_ELO and n2 >= MIN_MATCHES_FOR_ELO:
            training_rows.append({
                'match_id': match_id,
                'date': date,
                'player1': p1,
                'player2': p2,
                'surface': surface,
                'elo1': elo1,
                'elo2': elo2,
                'form_diff': form_diff,
                'rest_diff': rest_diff,
                'winner': winner,
            })
        
        # Update Elo after match
        k = SURFACE_K.get(surface, K_FACTOR)
        new_elo1, new_elo2 = update_elo(elo1, elo2, winner, k)
        player_elo[p1][surface] = new_elo1
        player_elo[p2][surface] = new_elo2
        
        # Update match counts and form
        player_matches[p1][surface] += 1
        player_matches[p2][surface] += 1
        player_form[p1][surface].append(winner)
        player_form[p2][surface].append(1 - winner)
    
    df = pd.DataFrame(training_rows)
    print(f"Training samples: {len(df)}")
    print(f"Winner distribution: {df['winner'].value_counts().to_dict()}")
    return df


def train_model(train_df):
    """Train and calibrate the match prediction model."""
    print("Training model...")
    
    # Features: elo1, elo2, form_diff, rest_diff
    # Target: winner (1 = player1 wins)
    X = train_df[['elo1', 'elo2', 'form_diff', 'rest_diff']].values
    y = train_df['winner'].values
    
    # Time-series split: use chronological order
    # Sort by date (already sorted)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Try multiple models
    models = {
        'logistic': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'rf': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, 
                                     class_weight='balanced', random_state=42, n_jobs=-1),
    }
    
    best_model = None
    best_score = float('inf')
    
    for name, model in models.items():
        # Calibrate probabilities
        calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
        calibrated.fit(X_train, y_train)
        
        # Evaluate
        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        print(f"{name}: Acc={acc:.4f}, LogLoss={ll:.4f}, Brier={brier:.4f}")
        
        if ll < best_score:
            best_score = ll
            best_model = calibrated
    
    # Final evaluation on test set
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    print(f"\nBest model test metrics:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Log Loss: {log_loss(y_test, y_prob):.4f}")
    print(f"  Brier Score: {brier_score_loss(y_test, y_prob):.4f}")
    
    # Feature importance (for RF)
    if hasattr(best_model.calibrated_classifiers_[0].estimator, 'feature_importances_'):
        importances = best_model.calibrated_classifiers_[0].estimator.feature_importances_
        for feat, imp in zip(['elo1', 'elo2', 'form_diff', 'rest_diff'], importances):
            print(f"  {feat}: {imp:.4f}")
    
    return best_model


def save_model(model, model_path):
    """Save model with metadata."""
    metadata = {
        'model_type': type(model).__name__,
        'features': ['elo1', 'elo2', 'form_diff', 'rest_diff'],
        'target': 'player1_wins',
        'elo_params': {
            'initial_elo': INITIAL_ELO,
            'k_factor': K_FACTOR,
            'surface_k': SURFACE_K,
        },
        'trained_at': datetime.now().isoformat(),
    }
    
    payload = {
        'model': model,
        'metadata': metadata,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")


def main():
    print("=" * 60)
    print("TENNIS ELO MODEL TRAINING")
    print("=" * 60)
    
    # Load matches
    matches, data_root = load_all_matches()
    
    # Build training data (this computes Elo and gets winners)
    train_df = build_training_data(matches, data_root)
    
    if len(train_df) < 100:
        print("ERROR: Not enough training data!")
        return
    
    # Train model
    model = train_model(train_df)
    
    # Save
    save_model(model, MODEL_PATH)
    
    print("\n✅ Training complete!")
    print(f"Model ready at: {MODEL_PATH}")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run src/app.py")


if __name__ == '__main__':
    main()