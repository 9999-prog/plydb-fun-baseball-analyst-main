"""
Build a game winner-prediction model.

Target (home_win) is derived directly from final scores in your raw
multiseason pitch-level file. Features are built from data you've already
computed: starting pitcher rolling form (home team's starter pitches to away
batters in the top of the 1st, and vice versa) + team batting strength
(average rolling hit rate across that game's batters) + park factor.

Input:  statcast_2023_2024_2025.parquet (raw, for final scores)
        statcast_multiseason_pa_level_model_ready.parquet (for features)
Output: ./win_model.joblib
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import joblib

RAW_PATH = "./data/pybaseball/statcast/statcast_2023_2024_2025.parquet"
PA_PATH = "./data/pybaseball/statcast/statcast_multiseason_pa_level_model_ready.parquet"
ADV_PATH = "./data/pybaseball/statcast/advanced_matchup_features.parquet"

pd.set_option("display.width", 120)

# ---------------------------------------------------------------
# 1. Derive the winner of each game from final scores
# ---------------------------------------------------------------
raw = pd.read_parquet(RAW_PATH, columns=[
    "game_pk", "game_date", "home_team", "away_team",
    "post_home_score", "post_away_score", "at_bat_number", "pitch_number",
])
raw = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"])
final = raw.groupby("game_pk").last().reset_index()
final["home_win"] = (final["post_home_score"] > final["post_away_score"]).astype(int)
final = final[["game_pk", "game_date", "home_team", "away_team", "home_win"]]
print(f"Derived a winner for {len(final):,} games")

# ---------------------------------------------------------------
# 2. Starting pitchers (home team's starter pitches in the top of the
#    1st against away batters; away team's starter pitches in the bottom)
# ---------------------------------------------------------------
pa_df = pd.read_parquet(PA_PATH)

pitcher_cols = ["pitcher", "pitcher_roll_k_rate", "pitcher_roll_bb_rate",
                "pitcher_roll_xba_against", "pitcher_roll_hardhit_against",
                "pitcher_roll_velo", "pitcher_roll_spin"]

home_starter = (
    pa_df[(pa_df["inning"] == 1) & (pa_df["inning_topbot"] == "Top")]
    .sort_values(["game_pk", "at_bat_number"]).groupby("game_pk").first()
    .reset_index()[["game_pk"] + pitcher_cols]
    .rename(columns={c: f"home_{c}" for c in pitcher_cols})
)
away_starter = (
    pa_df[(pa_df["inning"] == 1) & (pa_df["inning_topbot"] == "Bot")]
    .sort_values(["game_pk", "at_bat_number"]).groupby("game_pk").first()
    .reset_index()[["game_pk"] + pitcher_cols]
    .rename(columns={c: f"away_{c}" for c in pitcher_cols})
)

# ---------------------------------------------------------------
# 3. Team batting strength: average rolling hit rate/xBA among that
#    game's batters, split by which side of the inning they batted in
# ---------------------------------------------------------------
pa_df["batting_side"] = np.where(pa_df["inning_topbot"] == "Top", "away", "home")

# Use the first PA for each batter/game.  Later PAs can contain rolling values
# influenced by the same game's events; averaging all PAs would leak the game
# being predicted into the lineup-strength feature.
pregame_batters = (
    pa_df.sort_values(
        ["game_pk", "batting_side", "batter", "at_bat_number"],
        kind="mergesort",
    )
    .drop_duplicates(["game_pk", "batter"], keep="first")
)
team_strength = (
    pregame_batters.groupby(["game_pk", "batting_side"])
    .agg(avg_batter_hit_rate=("batter_roll_hit_rate", "mean"),
         avg_batter_xba=("batter_roll_xba", "mean"))
    .reset_index()
)
home_strength = (team_strength[team_strength["batting_side"] == "home"]
                  .drop(columns="batting_side")
                  .rename(columns={"avg_batter_hit_rate": "home_batting_hit_rate",
                                    "avg_batter_xba": "home_batting_xba"}))
away_strength = (team_strength[team_strength["batting_side"] == "away"]
                  .drop(columns="batting_side")
                  .rename(columns={"avg_batter_hit_rate": "away_batting_hit_rate",
                                    "avg_batter_xba": "away_batting_xba"}))

park = pa_df[["game_pk", "park_hit_factor", "park_hr_factor"]].drop_duplicates("game_pk")

# ---------------------------------------------------------------
# 4. Assemble the game-level table
# ---------------------------------------------------------------
game_df = (final.merge(home_starter, on="game_pk", how="left")
                 .merge(away_starter, on="game_pk", how="left")
                 .merge(home_strength, on="game_pk", how="left")
                 .merge(away_strength, on="game_pk", how="left")
                 .merge(park, on="game_pk", how="left"))
game_df["season"] = pd.to_datetime(game_df["game_date"]).dt.year

FEATURES = [
    "home_pitcher_roll_k_rate", "home_pitcher_roll_bb_rate", "home_pitcher_roll_xba_against",
    "home_pitcher_roll_hardhit_against", "home_pitcher_roll_velo", "home_pitcher_roll_spin",
    "away_pitcher_roll_k_rate", "away_pitcher_roll_bb_rate", "away_pitcher_roll_xba_against",
    "away_pitcher_roll_hardhit_against", "away_pitcher_roll_velo", "away_pitcher_roll_spin",
    "home_batting_hit_rate", "home_batting_xba", "away_batting_hit_rate", "away_batting_xba",
    "park_hit_factor", "park_hr_factor",
]
TARGET = "home_win"

model_df = game_df.dropna(subset=FEATURES + [TARGET]).copy()
print(f"Rows with complete features: {len(model_df):,} / {len(game_df):,}")

# ---------------------------------------------------------------
# 5. Evaluate: train on 2023+2024, test on entirely unseen 2025
# ---------------------------------------------------------------
train = model_df[model_df["season"] < 2025]
test = model_df[model_df["season"] == 2025]
print(f"Train: {len(train):,} | Test: {len(test):,}")

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

eval_model = XGBClassifier(
    n_estimators=300, max_depth=2, min_child_weight=1, reg_lambda=5,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
eval_model.fit(X_train, y_train)
test_preds = eval_model.predict_proba(X_test)[:, 1]

print(f"\nTest AUC:   {roc_auc_score(y_test, test_preds):.4f}")
print(f"Test Brier: {brier_score_loss(y_test, test_preds):.4f}")
print(f"Naive Brier (always predict {y_train.mean():.4f} home-win rate): "
      f"{brier_score_loss(y_test, np.full(len(y_test), y_train.mean())):.4f}")

importance_df = pd.DataFrame({
    "feature": FEATURES, "importance": eval_model.feature_importances_,
}).sort_values("importance", ascending=False)
print("\n--- Feature importance ---")
print(importance_df.to_string(index=False))

# ---------------------------------------------------------------
# 6. Retrain on ALL data and save the final model
# ---------------------------------------------------------------
X_all, y_all = model_df[FEATURES], model_df[TARGET]
final_model = XGBClassifier(
    n_estimators=300, max_depth=2, min_child_weight=1, reg_lambda=5,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
final_model.fit(X_all, y_all)
joblib.dump({"model": final_model, "features": FEATURES}, "./win_model.joblib")
print(f"\nFinal win model trained on {len(X_all):,} games -> ./win_model.joblib")