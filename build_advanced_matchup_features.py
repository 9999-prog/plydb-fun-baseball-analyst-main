"""Build leakage-aware, game-level matchup features.

The base PA pipeline does not currently create every optional advanced batting
field (for example xwOBA and HR rate).  This builder treats those fields as
optional and emits NaN rather than crashing.  It also uses the first PA for
each batter/game when aggregating lineup strength so later in-game events do
not leak into a pregame feature.

Run:
    python build_advanced_matchup_features.py
"""

import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
PA_PATH = os.path.join(
    ROOT,
    "data",
    "pybaseball",
    "statcast",
    "statcast_multiseason_pa_level_model_ready.parquet",
)
OUT_PATH = os.path.join(
    ROOT,
    "data",
    "pybaseball",
    "statcast",
    "advanced_matchup_features.parquet",
)

STARTER_FEATURES = [
    "pitcher_roll_k_rate",
    "pitcher_roll_bb_rate",
    "pitcher_roll_xba_against",
    "pitcher_roll_hardhit_against",
    "pitcher_roll_velo",
    "pitcher_roll_spin",
]
LINEUP_FEATURES = [
    "batter_roll_hit_rate",
    "batter_roll_xba",
    "batter_roll_xwoba",
    "batter_roll_hr_rate",
]


def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _normalise_team(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def _first_rows(df, keys, order_columns):
    order = [column for column in order_columns if column in df.columns]
    ordered = df.sort_values(order, kind="mergesort") if order else df
    return ordered.drop_duplicates(keys, keep="first")


def build_advanced_feature_table(pa_df):
    required = {
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "inning_topbot",
        "batter",
        "pitcher",
    }
    missing = sorted(required.difference(pa_df.columns))
    if missing:
        raise ValueError(f"Missing required PA columns: {', '.join(missing)}")

    df = pa_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_pk"] = safe_to_numeric(df["game_pk"])
    df["batter"] = safe_to_numeric(df["batter"])
    df["pitcher"] = safe_to_numeric(df["pitcher"])
    df["home_team"] = _normalise_team(df["home_team"])
    df["away_team"] = _normalise_team(df["away_team"])
    df["inning_topbot"] = _normalise_team(df["inning_topbot"]).str.lower()
    df["batting_team"] = np.where(
        df["inning_topbot"].eq("bot"), df["home_team"], df["away_team"]
    )
    df["pitching_team"] = np.where(
        df["inning_topbot"].eq("bot"), df["away_team"], df["home_team"]
    )

    events = (
        df["events"].fillna("").astype(str).str.lower()
        if "events" in df.columns
        else pd.Series("", index=df.index)
    )
    df["is_hit"] = events.isin({"single", "double", "triple", "home_run"}).astype(int)
    df["is_hr"] = events.eq("home_run").astype(int)
    df["is_walk"] = events.isin({"walk", "intent_walk"}).astype(int)
    df["is_strikeout"] = events.eq("strikeout").astype(int)

    if "bat_score" in df.columns and "post_bat_score" in df.columns:
        df["runs_on_pa"] = (
            safe_to_numeric(df["post_bat_score"])
            - safe_to_numeric(df["bat_score"])
        ).clip(lower=0).fillna(0)
    else:
        df["runs_on_pa"] = 0.0

    # These are optional extensions.  The base pipeline supplies the pitcher,
    # batter, and park fields but not xwOBA/HR rolling fields yet.
    for column in STARTER_FEATURES + LINEUP_FEATURES + [
        "park_hit_factor",
        "park_hr_factor",
        "p_throws",
        "inning",
        "at_bat_number",
    ]:
        if column not in df.columns:
            df[column] = np.nan

    # Identify starters by the first pitcher to appear for each team in game
    # order.  Sorting is important because parquet row order is not a contract.
    starters = (
        _first_rows(
            df.dropna(subset=["game_pk", "pitcher"]),
            ["game_pk", "pitching_team"],
            ["game_pk", "game_date", "inning", "at_bat_number"],
        )[["game_pk", "pitching_team", "pitcher"]]
        .rename(columns={"pitcher": "starter_id"})
    )
    df = df.merge(starters, on=["game_pk", "pitching_team"], how="left")
    df["is_bullpen_pa"] = (
        df["pitcher"].notna()
        & df["starter_id"].notna()
        & df["pitcher"].ne(df["starter_id"])
    )

    starter_rows = df[df["pitcher"].eq(df["starter_id"])].copy()
    starter_team_stats = (
        starter_rows.groupby(["game_pk", "pitching_team"], as_index=False)
        .agg(
            starter_id=("starter_id", "first"),
            starter_k_rate=("pitcher_roll_k_rate", "mean"),
            starter_bb_rate=("pitcher_roll_bb_rate", "mean"),
            starter_xba_allowed=("pitcher_roll_xba_against", "mean"),
            starter_hardhit_allowed=("pitcher_roll_hardhit_against", "mean"),
            starter_velo=("pitcher_roll_velo", "mean"),
            starter_spin=("pitcher_roll_spin", "mean"),
        )
    )

    bullpen_stats = (
        df[df["is_bullpen_pa"]]
        .groupby(["game_pk", "pitching_team"], as_index=False)
        .agg(
            bullpen_runs=("runs_on_pa", "sum"),
            bullpen_pa=("runs_on_pa", "size"),
        )
    )
    bullpen_stats["bullpen_rpa"] = (
        bullpen_stats["bullpen_runs"]
        / bullpen_stats["bullpen_pa"].clip(lower=1)
    )
    team_game_stats = starter_team_stats.merge(
        bullpen_stats[["game_pk", "pitching_team", "bullpen_rpa"]],
        on=["game_pk", "pitching_team"],
        how="left",
    ).rename(columns={"pitching_team": "team"})

    game_summary = _first_rows(
        df[["game_pk", "game_date", "home_team", "away_team"]].dropna(
            subset=["game_pk"]
        ),
        ["game_pk"],
        ["game_pk", "game_date"],
    )

    home_stats = team_game_stats.rename(
        columns={
            "team": "home_team",
            "starter_id": "home_starter_id",
            "starter_k_rate": "home_starter_k_rate",
            "starter_bb_rate": "home_starter_bb_rate",
            "starter_xba_allowed": "home_starter_xba_allowed",
            "starter_hardhit_allowed": "home_starter_hardhit_allowed",
            "starter_velo": "home_starter_velo",
            "starter_spin": "home_starter_spin",
            "bullpen_rpa": "home_bullpen_rpa",
        }
    )
    away_stats = team_game_stats.rename(
        columns={
            "team": "away_team",
            "starter_id": "away_starter_id",
            "starter_k_rate": "away_starter_k_rate",
            "starter_bb_rate": "away_starter_bb_rate",
            "starter_xba_allowed": "away_starter_xba_allowed",
            "starter_hardhit_allowed": "away_starter_hardhit_allowed",
            "starter_velo": "away_starter_velo",
            "starter_spin": "away_starter_spin",
            "bullpen_rpa": "away_bullpen_rpa",
        }
    )
    game_features = game_summary.merge(
        home_stats, on=["game_pk", "home_team"], how="left"
    ).merge(away_stats, on=["game_pk", "away_team"], how="left")

    # Use one pregame row per batter/game.  Averaging every PA would allow a
    # batter's own hits/walks later in the game to affect the same game's
    # lineup-strength feature.
    batter_game = _first_rows(
        df.dropna(subset=["game_pk", "batter"]),
        ["game_pk", "batter"],
        ["game_pk", "game_date", "at_bat_number"],
    )
    lineup = (
        batter_game.groupby(["game_pk", "batting_team"], as_index=False)
        .agg(
            lineup_hit_rate=("batter_roll_hit_rate", "mean"),
            lineup_xba=("batter_roll_xba", "mean"),
            lineup_xwoba=("batter_roll_xwoba", "mean"),
            lineup_hr_rate=("batter_roll_hr_rate", "mean"),
        )
    )
    game_features = game_features.merge(
        lineup.rename(
            columns={
                "batting_team": "home_team",
                "lineup_hit_rate": "home_lineup_hit_rate",
                "lineup_xba": "home_lineup_xba",
                "lineup_xwoba": "home_lineup_xwoba",
                "lineup_hr_rate": "home_lineup_hr_rate",
            }
        ),
        on=["game_pk", "home_team"],
        how="left",
    ).merge(
        lineup.rename(
            columns={
                "batting_team": "away_team",
                "lineup_hit_rate": "away_lineup_hit_rate",
                "lineup_xba": "away_lineup_xba",
                "lineup_xwoba": "away_lineup_xwoba",
                "lineup_hr_rate": "away_lineup_hr_rate",
            }
        ),
        on=["game_pk", "away_team"],
        how="left",
    )

    first_game_row = _first_rows(
        df,
        ["game_pk"],
        ["game_pk", "game_date", "at_bat_number"],
    )
    park = first_game_row[
        ["game_pk", "home_team", "park_hit_factor", "park_hr_factor"]
    ]
    game_features = game_features.merge(
        park.rename(
            columns={
                "park_hit_factor": "home_park_hit_factor",
                "park_hr_factor": "home_park_hr_factor",
            }
        ),
        on=["game_pk", "home_team"],
        how="left",
    )

    starter_hands = (
        starter_rows.dropna(subset=["p_throws"])
        .groupby(["game_pk", "pitching_team"], as_index=False)["p_throws"]
        .first()
        .rename(columns={"pitching_team": "team", "p_throws": "starter_throws"})
    )
    game_features = game_features.merge(
        starter_hands.rename(
            columns={"team": "home_team", "starter_throws": "home_starter_throws"}
        ),
        on=["game_pk", "home_team"],
        how="left",
    ).merge(
        starter_hands.rename(
            columns={"team": "away_team", "starter_throws": "away_starter_throws"}
        ),
        on=["game_pk", "away_team"],
        how="left",
    )

    game_features = game_features.drop_duplicates(
        subset=["game_pk", "home_team", "away_team"]
    ).reset_index(drop=True)
    return game_features


def main():
    if not os.path.exists(PA_PATH):
        raise FileNotFoundError(f"PA-level Statcast file not found: {PA_PATH}")

    pa_df = pd.read_parquet(PA_PATH)
    print(f"Loaded {len(pa_df):,} PA rows")
    features = build_advanced_feature_table(pa_df)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    features.to_parquet(OUT_PATH, index=False)
    print(f"Saved advanced matchup features to {OUT_PATH}")
    print(features.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
