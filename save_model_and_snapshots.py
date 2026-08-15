"""
Finalize the model for real use:
  1. Retrain XGBoost on ALL available data (not just the train split) using
     the best hyperparameters found during tuning -- once you're done
     evaluating, using every row you have makes for the strongest model.
  2. Save the trained model to disk.
  3. Build "current form" snapshot tables: each batter's and pitcher's most
     recent rolling stats AS OF RIGHT NOW (not shifted -- this represents
     what you'd know walking into a game that hasn't happened yet).
  4. Save park factors as a lookup table.

Run this ONCE after you're happy with tuning. Re-run it whenever you want to
refresh "current form" with new games (e.g. after adding 2024 data).
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# 1. Load full game-level dataset
# ---------------------------------------------------------------
df = pd.read_parquet("./data/pybaseball/statcast/statcast_multiseason_batter_game_level.parquet")

FEATURES = [
    "batter_roll_hit_rate", "batter_roll_xba", "batter_roll_hardhit_rate",
    "batter_roll_k_rate", "batter_roll_bb_rate", "batter_roll_avg_ev",
    "batter_roll_hit_rate_vs_hand", "batter_roll_xba_vs_hand", "platoon_advantage",
    "pitcher_roll_k_rate", "pitcher_roll_bb_rate", "pitcher_roll_xba_against",
    "pitcher_roll_hardhit_against", "pitcher_roll_velo", "pitcher_roll_spin",
    "park_hit_factor", "park_hr_factor",
]
TARGET = "hit_in_game"

model_df = df.dropna(subset=FEATURES + [TARGET]).copy()
X, y = model_df[FEATURES], model_df[TARGET]

# ---------------------------------------------------------------
# 2. Retrain on ALL data with the tuned hyperparameters
#    (update these if your tuning run found different best_params_)
# ---------------------------------------------------------------
BEST_PARAMS = dict(max_depth=2, min_child_weight=1, reg_lambda=5)

final_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    **BEST_PARAMS,
)
final_model.fit(X, y)
print(f"Final model trained on {len(X):,} batter-games (all available data)")

# ---------------------------------------------------------------
# 3. Save the model + the feature list it expects (so predict script
#    always builds features in the exact right order)
# ---------------------------------------------------------------
joblib.dump({"model": final_model, "features": FEATURES}, "./hit_model.joblib")
print("Saved -> ./hit_model.joblib")

# ---------------------------------------------------------------
# 4. Build "current form" snapshots -- most recent games' worth of
#    stats for each batter and pitcher, computed WITHOUT shifting
#    (this reflects form entering a game that hasn't happened yet)
# ---------------------------------------------------------------
pa_df = pd.read_parquet("./data/pybaseball/statcast/statcast_multiseason_pa_level_model_ready.parquet")
pa_df = pa_df.sort_values(["batter", "game_date", "game_pk", "at_bat_number"])

BATTER_WINDOW = 25
batter_snapshot = (
    pa_df.groupby("batter")
    .tail(BATTER_WINDOW)
    .groupby("batter")
    .agg(
        batter_roll_hit_rate=("is_hit", "mean"),
        batter_roll_xba=("estimated_ba_using_speedangle", "mean"),
        batter_roll_hardhit_rate=("is_hard_hit" if "is_hard_hit" in pa_df.columns else "is_hit", "mean"),
        batter_roll_k_rate=("is_strikeout", "mean"),
        batter_roll_bb_rate=("is_walk", "mean"),
        batter_roll_avg_ev=("launch_speed", "mean"),
    )
    .reset_index()
)

PLATOON_WINDOW = 40
platoon_snapshot = (
    pa_df.groupby(["batter", "p_throws"])
    .tail(PLATOON_WINDOW)
    .groupby(["batter", "p_throws"])
    .agg(
        batter_roll_hit_rate_vs_hand=("is_hit", "mean"),
        batter_roll_xba_vs_hand=("estimated_ba_using_speedangle", "mean"),
    )
    .reset_index()
)

pitcher_game = (
    pa_df.groupby(["pitcher", "game_pk", "game_date"])
    .agg(
        pa_faced=("is_at_bat", "sum"),
        strikeouts=("is_strikeout", "sum"),
        walks_allowed=("is_walk", "sum"),
        avg_xba_against=("estimated_ba_using_speedangle", "mean"),
        avg_velo=("avg_release_speed", "mean"),
        avg_spin=("avg_spin_rate", "mean"),
    )
    .reset_index()
)
pitcher_game["k_rate"] = pitcher_game["strikeouts"] / pitcher_game["pa_faced"]
pitcher_game["bb_rate"] = pitcher_game["walks_allowed"] / pitcher_game["pa_faced"]
pitcher_game = pitcher_game.sort_values(["pitcher", "game_date"])

PITCHER_WINDOW = 5
pitcher_snapshot = (
    pitcher_game.groupby("pitcher")
    .tail(PITCHER_WINDOW)
    .groupby("pitcher")
    .agg(
        pitcher_roll_k_rate=("k_rate", "mean"),
        pitcher_roll_bb_rate=("bb_rate", "mean"),
        pitcher_roll_xba_against=("avg_xba_against", "mean"),
        pitcher_roll_velo=("avg_velo", "mean"),
        pitcher_roll_spin=("avg_spin", "mean"),
    )
    .reset_index()
)
# hardhit_against needs its own pass since it wasn't in pitcher_game above
pitcher_game2 = (
    pa_df.groupby(["pitcher", "game_pk"])
    .agg(hard_hits=("is_hard_hit", "sum"), balls_in_play=("launch_speed", "count"))
    .reset_index()
)
pitcher_game2["hardhit_rate_against"] = pitcher_game2["hard_hits"] / pitcher_game2["balls_in_play"].replace(0, np.nan)
pitcher_hardhit_snapshot = (
    pitcher_game2.groupby("pitcher").tail(PITCHER_WINDOW)
    .groupby("pitcher").agg(pitcher_roll_hardhit_against=("hardhit_rate_against", "mean"))
    .reset_index()
)
pitcher_snapshot = pitcher_snapshot.merge(pitcher_hardhit_snapshot, on="pitcher", how="left")

park_snapshot = pa_df[["home_team", "park_hit_factor", "park_hr_factor"]].drop_duplicates()

joblib.dump({
    "batter_snapshot": batter_snapshot,
    "platoon_snapshot": platoon_snapshot,
    "pitcher_snapshot": pitcher_snapshot,
    "park_snapshot": park_snapshot,
}, "./current_form_snapshots.joblib")

print("Saved -> ./current_form_snapshots.joblib")
print(f"  {len(batter_snapshot):,} batters, {len(pitcher_snapshot):,} pitchers, "
      f"{len(park_snapshot):,} parks")