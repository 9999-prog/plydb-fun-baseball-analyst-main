"""
FULL PIPELINE: raw pitch-level Statcast data -> game-level, model-ready table.
Works for a single season or multiple seasons combined into one file.

USAGE:
    Change RAW_INPUT_PATH below to point at your combined parquet
    (e.g. all of 2023+2024+2025 pitches in one file), then run:
        python run_full_pipeline.py

    To add a new season later: combine the new season's files into your
    raw pitch-level parquet the same way you did for 2023 (your existing
    combine_data.py script), point RAW_INPUT_PATH at the new combined file,
    and re-run this ONE script. No need to touch the 5 old scripts again.

WHAT THIS DOES (in order):
    1. Aggregate pitches -> plate appearances (PA-level, including score deltas)
    2. Add rolling batter form features
    3. Add rolling pitcher form features
    4. Add matchup context (platoon splits, park factors)
    5. Roll up to batter-game level with the hit_in_game target

SEASON-AWARE ROLLING: all rolling windows are grouped by (player, season) so
a player's trailing stats reset each year instead of carrying over from the
prior season. This is a deliberate design choice -- see the note in step 2.
"""

import pandas as pd
import numpy as np

# =================================================================
# CONFIG -- change this line when you add new seasons
# =================================================================
RAW_INPUT_PATH = "./data/pybaseball/statcast/statcast_2023_2024_2025.parquet"
OUTPUT_PREFIX = "./data/pybaseball/statcast/statcast_multiseason"

BATTER_ROLL_WINDOW = 25
BATTER_MIN_PERIODS = 5
PITCHER_ROLL_WINDOW = 5
PITCHER_MIN_PERIODS = 2
PLATOON_ROLL_WINDOW = 40
PLATOON_MIN_PERIODS = 8
PARK_MIN_PRIOR_PA = 200


# =================================================================
# STEP 1: pitch-level -> PA-level
# =================================================================
def aggregate_pa_level(raw_path):
    df = pd.read_parquet(raw_path)
    print(f"[1/5] Loaded {len(df):,} pitches, {df.shape[1]} columns")

    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    terminal = df[df["events"].notna()].copy()

    HIT_EVENTS = {"single", "double", "triple", "home_run"}
    AB_EXCLUDE = {
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "catcher_interf", "sac_fly_double_play", "sac_bunt_double_play",
    }
    terminal["is_hit"] = terminal["events"].isin(HIT_EVENTS).astype(int)
    terminal["is_at_bat"] = (~terminal["events"].isin(AB_EXCLUDE)).astype(int)
    terminal["is_walk"] = (terminal["events"] == "walk").astype(int)
    terminal["is_strikeout"] = (terminal["events"] == "strikeout").astype(int)
    terminal["is_hbp"] = (terminal["events"] == "hit_by_pitch").astype(int)

    contact_cols = ["launch_speed", "launch_angle", "estimated_ba_using_speedangle",
                     "estimated_woba_using_speedangle", "bb_type", "hit_distance_sc"]
    contact_cols = [c for c in contact_cols if c in terminal.columns]

    keep_cols = [
        "game_pk", "game_date", "season", "at_bat_number",
        "batter", "pitcher", "stand", "p_throws",
        "home_team", "away_team", "inning", "inning_topbot",
        "balls", "strikes", "outs_when_up",
        # Keep terminal score fields so downstream bullpen/form/total models
        # can reconstruct runs scored on each plate appearance.
        "bat_score", "post_bat_score", "post_home_score", "post_away_score",
        "events", "description",
        "is_hit", "is_at_bat", "is_walk", "is_strikeout", "is_hbp",
    ] + contact_cols
    keep_cols = [c for c in keep_cols if c in terminal.columns]
    pa_df = terminal[keep_cols].reset_index(drop=True)
    if "bat_score" in pa_df.columns and "post_bat_score" in pa_df.columns:
        pa_df["runs_on_pa"] = (
            pd.to_numeric(pa_df["post_bat_score"], errors="coerce")
            - pd.to_numeric(pa_df["bat_score"], errors="coerce")
        ).clip(lower=0).fillna(0.0)
    else:
        # A missing score feed is safe but should be visible to downstream
        # coverage checks rather than silently inventing runs.
        pa_df["runs_on_pa"] = 0.0

    pitch_summary = (
        df.groupby(["game_pk", "at_bat_number"])
        .agg(
            n_pitches=("pitch_number", "max"),
            avg_release_speed=("release_speed", "mean"),
            max_release_speed=("release_speed", "max"),
            avg_spin_rate=("release_spin_rate", "mean"),
        )
        .reset_index()
    )
    pa_df = pa_df.merge(pitch_summary, on=["game_pk", "at_bat_number"], how="left")

    print(f"[1/5] -> {len(pa_df):,} plate appearances")
    return pa_df


# =================================================================
# STEP 2: rolling batter features (season-aware)
# =================================================================
def add_batter_features(pa_df):
    pa_df = pa_df.sort_values(["batter", "game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)
    pa_df["is_hard_hit"] = (pa_df["launch_speed"] >= 95).astype(float)

    def roll(df, col, window, min_periods):
        # group by (batter, season) so windows reset at season boundaries
        return (
            df.groupby(["batter", "season"])[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        )

    pa_df["batter_roll_hit_rate"]     = roll(pa_df, "is_hit", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)
    pa_df["batter_roll_xba"]          = roll(pa_df, "estimated_ba_using_speedangle", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)
    pa_df["batter_roll_hardhit_rate"] = roll(pa_df, "is_hard_hit", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)
    pa_df["batter_roll_k_rate"]       = roll(pa_df, "is_strikeout", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)
    pa_df["batter_roll_bb_rate"]      = roll(pa_df, "is_walk", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)
    pa_df["batter_roll_avg_ev"]       = roll(pa_df, "launch_speed", BATTER_ROLL_WINDOW, BATTER_MIN_PERIODS)

    print(f"[2/5] Batter rolling features added (season-aware, window={BATTER_ROLL_WINDOW})")
    return pa_df


# =================================================================
# STEP 3: rolling pitcher features (season-aware, per-appearance)
# =================================================================
def add_pitcher_features(pa_df):
    pitcher_game = (
        pa_df.groupby(["pitcher", "season", "game_pk", "game_date"])
        .agg(
            pa_faced=("is_at_bat", "count"),
            hits_allowed=("is_hit", "sum"),
            walks_allowed=("is_walk", "sum"),
            strikeouts=("is_strikeout", "sum"),
            hard_hits_allowed=("is_hard_hit", "sum"),
            balls_in_play=("launch_speed", "count"),
            avg_xba_against=("estimated_ba_using_speedangle", "mean"),
            avg_velo=("avg_release_speed", "mean"),
            avg_spin=("avg_spin_rate", "mean"),
        )
        .reset_index()
    )
    pitcher_game["k_rate"] = pitcher_game["strikeouts"] / pitcher_game["pa_faced"]
    pitcher_game["bb_rate"] = pitcher_game["walks_allowed"] / pitcher_game["pa_faced"]
    pitcher_game["hardhit_rate_against"] = (
        pitcher_game["hard_hits_allowed"] / pitcher_game["balls_in_play"].replace(0, np.nan)
    )

    pitcher_game = pitcher_game.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)

    def roll(df, col, window, min_periods):
        return (
            df.groupby(["pitcher", "season"])[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
        )

    pitcher_game["pitcher_roll_k_rate"]          = roll(pitcher_game, "k_rate", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)
    pitcher_game["pitcher_roll_bb_rate"]         = roll(pitcher_game, "bb_rate", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)
    pitcher_game["pitcher_roll_xba_against"]     = roll(pitcher_game, "avg_xba_against", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)
    pitcher_game["pitcher_roll_hardhit_against"] = roll(pitcher_game, "hardhit_rate_against", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)
    pitcher_game["pitcher_roll_velo"]            = roll(pitcher_game, "avg_velo", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)
    pitcher_game["pitcher_roll_spin"]            = roll(pitcher_game, "avg_spin", PITCHER_ROLL_WINDOW, PITCHER_MIN_PERIODS)

    pitcher_features = pitcher_game[[
        "pitcher", "game_pk",
        "pitcher_roll_k_rate", "pitcher_roll_bb_rate", "pitcher_roll_xba_against",
        "pitcher_roll_hardhit_against", "pitcher_roll_velo", "pitcher_roll_spin",
    ]]
    pa_df = pa_df.merge(pitcher_features, on=["pitcher", "game_pk"], how="left")

    print(f"[3/5] Pitcher rolling features added ({len(pitcher_game):,} pitcher-appearances, window={PITCHER_ROLL_WINDOW})")
    return pa_df


# =================================================================
# STEP 4: matchup context (platoon splits + data-driven park factors)
# =================================================================
def add_matchup_features(df):
    df = df.sort_values(["batter", "game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)

    df["batter_roll_hit_rate_vs_hand"] = (
        df.groupby(["batter", "season", "p_throws"])["is_hit"]
        .transform(lambda s: s.shift(1).rolling(PLATOON_ROLL_WINDOW, min_periods=PLATOON_MIN_PERIODS).mean())
    )
    df["batter_roll_xba_vs_hand"] = (
        df.groupby(["batter", "season", "p_throws"])["estimated_ba_using_speedangle"]
        .transform(lambda s: s.shift(1).rolling(PLATOON_ROLL_WINDOW, min_periods=PLATOON_MIN_PERIODS).mean())
    )
    df["platoon_advantage"] = (df["stand"] != df["p_throws"]).astype(int)

    # Park factors must be known before the game being scored.  The old
    # implementation used the final full-season rate, which leaked future
    # outcomes into every training row.  Build game-level cumulative rates and
    # shift them by one game; early-season games use a neutral factor.
    chronological = df.sort_values(
        ["season", "game_date", "game_pk", "at_bat_number"],
        kind="mergesort",
    )
    park_games = (
        chronological.groupby(
            ["home_team", "season", "game_pk", "game_date"],
            as_index=False,
        )
        .agg(
            park_hits=("is_hit", "sum"),
            park_pa=("is_hit", "size"),
            park_home_runs=("events", lambda s: (s == "home_run").sum()),
        )
        .sort_values(["season", "game_date", "game_pk"], kind="mergesort")
    )

    def prior_cumulative(column, groups):
        return park_games.groupby(groups)[column].transform(
            lambda s: s.shift(1).cumsum()
        )

    park_groups = ["home_team", "season"]
    league_groups = ["season"]
    park_games["prior_park_hits"] = prior_cumulative("park_hits", park_groups)
    park_games["prior_park_pa"] = prior_cumulative("park_pa", park_groups)
    park_games["prior_park_home_runs"] = prior_cumulative(
        "park_home_runs", park_groups
    )
    park_games["prior_lg_hits"] = prior_cumulative("park_hits", league_groups)
    park_games["prior_lg_pa"] = prior_cumulative("park_pa", league_groups)
    park_games["prior_lg_home_runs"] = prior_cumulative(
        "park_home_runs", league_groups
    )

    park_hit_rate = park_games["prior_park_hits"] / park_games["prior_park_pa"].replace(0, np.nan)
    league_hit_rate = park_games["prior_lg_hits"] / park_games["prior_lg_pa"].replace(0, np.nan)
    park_hr_rate = park_games["prior_park_home_runs"] / park_games["prior_park_pa"].replace(0, np.nan)
    league_hr_rate = park_games["prior_lg_home_runs"] / park_games["prior_lg_pa"].replace(0, np.nan)

    enough_history = (
        park_games["prior_park_pa"].ge(PARK_MIN_PRIOR_PA)
        & park_games["prior_lg_pa"].ge(PARK_MIN_PRIOR_PA)
    )
    park_games["park_hit_factor"] = (
        (park_hit_rate / league_hit_rate).where(enough_history, 1.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    park_games["park_hr_factor"] = (
        (park_hr_rate / league_hr_rate).where(enough_history, 1.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )

    df = df.merge(
        park_games[[
            "home_team", "season", "game_pk", "park_hit_factor", "park_hr_factor"
        ]],
        on=["home_team", "season", "game_pk"],
        how="left",
    )

    print("[4/5] Matchup features added (platoon splits + pregame park factors)")
    return df


# =================================================================
# STEP 5: roll up to batter-game level with hit_in_game target
# =================================================================
def build_game_level(df):
    df = df.sort_values(["batter", "game_pk", "at_bat_number"])

    game_target = (
        df.groupby(["batter", "game_pk"])["is_hit"].max()
        .rename("hit_in_game").reset_index()
    )
    pa_count = (
        df.groupby(["batter", "game_pk"]).size()
        .rename("pa_count_this_game").reset_index()
    )
    first_pa = df.groupby(["batter", "game_pk"]).first().reset_index()

    feature_cols = [
        "batter", "pitcher", "game_pk", "game_date", "season", "home_team", "away_team",
        "stand", "p_throws",
        "batter_roll_hit_rate", "batter_roll_xba", "batter_roll_hardhit_rate",
        "batter_roll_k_rate", "batter_roll_bb_rate", "batter_roll_avg_ev",
        "batter_roll_hit_rate_vs_hand", "batter_roll_xba_vs_hand", "platoon_advantage",
        "pitcher_roll_k_rate", "pitcher_roll_bb_rate", "pitcher_roll_xba_against",
        "pitcher_roll_hardhit_against", "pitcher_roll_velo", "pitcher_roll_spin",
        "park_hit_factor", "park_hr_factor",
    ]
    feature_cols = [c for c in feature_cols if c in first_pa.columns]

    game_df = first_pa[feature_cols].merge(game_target, on=["batter", "game_pk"])
    game_df = game_df.merge(pa_count, on=["batter", "game_pk"])

    print(f"[5/5] -> {len(game_df):,} batter-games, base rate {game_df['hit_in_game'].mean():.4f}")
    return game_df


# =================================================================
# RUN THE FULL PIPELINE
# =================================================================
if __name__ == "__main__":
    pa_df = aggregate_pa_level(RAW_INPUT_PATH)
    pa_df = add_batter_features(pa_df)
    pa_df = add_pitcher_features(pa_df)
    pa_df = add_matchup_features(pa_df)

    pa_out = f"{OUTPUT_PREFIX}_pa_level_model_ready.parquet"
    pa_df.to_parquet(pa_out, index=False)
    print(f"Saved PA-level -> {pa_out}")

    game_df = build_game_level(pa_df)
    game_out = f"{OUTPUT_PREFIX}_batter_game_level.parquet"
    game_df.to_parquet(game_out, index=False)
    print(f"Saved game-level -> {game_out}")

    print("\nPipeline complete. By season:")
    print(game_df.groupby("season")["hit_in_game"].agg(["count", "mean"]))