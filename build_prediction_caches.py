"""
MLB PREDICTION CACHE BUILDER - FIXED / EXPANDED

Creates:

    team_season_stats.joblib
    h2h_stats.joblib
    pitcher_matchups.joblib
    batter_matchups.joblib

Uses the existing Statcast parquet.

Important:
- Handles pandas NA values safely.
- Does NOT crash on nullable boolean columns.
- Builds season-long team batting statistics.
- Builds team-vs-team historical H2H statistics.
- Builds pitcher-vs-opponent-team statistics.
- Builds batter-vs-pitcher historical statistics.
- Adds recent-form fields where possible.
- Saves everything locally for FAST prediction mode.

Run:

    python build_prediction_caches.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

STATCAST_FILE = os.path.join(
    ROOT,
    "data",
    "pybaseball",
    "statcast",
    "statcast_multiseason_pa_level_model_ready.parquet"
)

print("\n" + "=" * 80)
print("MLB PREDICTION CACHE BUILDER")
print("=" * 80)


# ================================================================
# LOAD STATCAST
# ================================================================

if not os.path.exists(STATCAST_FILE):

    raise FileNotFoundError(
        "\nStatcast file not found:\n"
        f"{STATCAST_FILE}"
    )


print("\nLoading Statcast data...")

df = pd.read_parquet(
    STATCAST_FILE
)

print(
    f"Loaded {len(df):,} plate appearances."
)


# ================================================================
# REQUIRED COLUMNS
# ================================================================

required = [
    "game_date",
    "game_pk",
    "home_team",
    "away_team",
    "inning_topbot",
    "batter",
    "pitcher",
    "events"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        "\nMissing required columns:\n"
        + "\n".join(missing)
    )


# ================================================================
# NORMALISE
# ================================================================

df["game_date"] = pd.to_datetime(
    df["game_date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "game_date",
        "game_pk",
        "batter",
        "pitcher"
    ]
).copy()


# ================================================================
# TEAM IDENTIFICATION
# ================================================================

inning = (
    df["inning_topbot"]
    .astype("string")
    .str.lower()
    .str.strip()
)

df["batting_team"] = np.where(
    inning.eq("bot"),
    df["home_team"],
    df["away_team"]
)

df["opposing_team"] = np.where(
    df["batting_team"] == df["home_team"],
    df["away_team"],
    df["home_team"]
)


# ================================================================
# SEASON
# ================================================================

df["season"] = (
    df["game_date"]
    .dt.year
)


CURRENT_SEASON = int(
    df["season"].max()
)

print(
    f"Current Statcast season: {CURRENT_SEASON}"
)


# ================================================================
# EVENT NORMALISATION
# ================================================================

df["event_clean"] = (
    df["events"]
    .astype("string")
    .str.lower()
    .str.strip()
)


# ================================================================
# SAFE FLAGS
#
# IMPORTANT:
# We explicitly fill missing values before converting to int.
# This fixes:
#
# ValueError: cannot convert NA to integer
# ================================================================

hit_events = {
    "single",
    "double",
    "triple",
    "home_run"
}

walk_events = {
    "walk",
    "intent_walk"
}

strikeout_events = {
    "strikeout"
}


df["is_hit"] = (
    df["event_clean"]
    .isin(hit_events)
    .fillna(False)
    .astype("int8")
)


df["is_hr"] = (
    df["event_clean"]
    .eq("home_run")
    .fillna(False)
    .astype("int8")
)


df["is_walk"] = (
    df["event_clean"]
    .isin(walk_events)
    .fillna(False)
    .astype("int8")
)


df["is_strikeout"] = (
    df["event_clean"]
    .isin(strikeout_events)
    .fillna(False)
    .astype("int8")
)


# ================================================================
# NUMERIC STATCAST COLUMNS
# ================================================================

numeric_columns = [
    "launch_speed",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "launch_angle"
]

for col in numeric_columns:

    if col not in df.columns:

        df[col] = np.nan

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ================================================================
# HARD HIT
#
# SAFE AGAINST NA VALUES
# ================================================================

df["hard_hit"] = (
    df["launch_speed"]
    .ge(95)
    .fillna(False)
    .astype("int8")
)


# ================================================================
# GAME TABLE
# ================================================================

games = (
    df[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team"
        ]
    ]
    .drop_duplicates("game_pk")
    .copy()
)

print(
    f"Unique games available: {len(games):,}"
)


# ================================================================
# 1. TEAM SEASON STATS
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "BUILDING TEAM SEASON STATISTICS"
)

print(
    "=" * 80
)


season_df = df[
    df["season"] == CURRENT_SEASON
].copy()


team_rows = []


for team, group in season_df.groupby(
    "batting_team"
):

    pa = len(group)

    hits = int(
        group["is_hit"].sum()
    )

    hrs = int(
        group["is_hr"].sum()
    )

    walks = int(
        group["is_walk"].sum()
    )

    strikeouts = int(
        group["is_strikeout"].sum()
    )

    team_rows.append(
        {
            "team":
                team,

            "season":
                CURRENT_SEASON,

            "plate_appearances":
                pa,

            "hits":
                hits,

            "home_runs":
                hrs,

            "walks":
                walks,

            "strikeouts":
                strikeouts,

            "hit_rate":
                hits / pa
                if pa
                else np.nan,

            "walk_rate":
                walks / pa
                if pa
                else np.nan,

            "strikeout_rate":
                strikeouts / pa
                if pa
                else np.nan,

            "hard_hit_rate":
                group[
                    "hard_hit"
                ].mean(),

            "xBA":
                group[
                    "estimated_ba_using_speedangle"
                ].mean(),

            "xwOBA":
                group[
                    "estimated_woba_using_speedangle"
                ].mean(),

            "exit_velocity":
                group[
                    "launch_speed"
                ].mean()
        }
    )


team_season = pd.DataFrame(
    team_rows
)


joblib.dump(
    team_season,
    os.path.join(
        ROOT,
        "team_season_stats.joblib"
    )
)


print(
    f"Saved team_season_stats.joblib "
    f"({len(team_season)} teams)"
)


# ================================================================
# 2. TEAM H2H
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "BUILDING TEAM H2H DATABASE"
)

print(
    "=" * 80
)


h2h_rows = []


for _, game in games.iterrows():

    game_id = game[
        "game_pk"
    ]

    home = game[
        "home_team"
    ]

    away = game[
        "away_team"
    ]

    matchup = df[
        df["game_pk"] == game_id
    ]

    if matchup.empty:

        continue


    for team, opponent, is_home in [
        (home, away, True),
        (away, home, False)
    ]:

        batting = matchup[
            matchup["batting_team"]
            == team
        ]

        if batting.empty:

            continue


        h2h_rows.append(
            {
                "game_pk":
                    game_id,

                "game_date":
                    game["game_date"],

                "season":
                    game["game_date"].year,

                "team":
                    team,

                "opponent":
                    opponent,

                "home":
                    is_home,

                "team_pa":
                    len(batting),

                "team_hits":
                    int(
                        batting[
                            "is_hit"
                        ].sum()
                    ),

                "team_home_runs":
                    int(
                        batting[
                            "is_hr"
                        ].sum()
                    ),

                "team_walks":
                    int(
                        batting[
                            "is_walk"
                        ].sum()
                    ),

                "team_strikeouts":
                    int(
                        batting[
                            "is_strikeout"
                        ].sum()
                    ),

                "team_hit_rate":
                    batting[
                        "is_hit"
                    ].mean(),

                "team_xba":
                    batting[
                        "estimated_ba_using_speedangle"
                    ].mean(),

                "team_xwoba":
                    batting[
                        "estimated_woba_using_speedangle"
                    ].mean(),

                "team_exit_velocity":
                    batting[
                        "launch_speed"
                    ].mean(),

                "team_hard_hit_rate":
                    batting[
                        "hard_hit"
                    ].mean()
            }
        )


h2h = pd.DataFrame(
    h2h_rows
)


joblib.dump(
    h2h,
    os.path.join(
        ROOT,
        "h2h_stats.joblib"
    )
)


print(
    f"Saved h2h_stats.joblib "
    f"({len(h2h):,} team-game records)"
)


# ================================================================
# 3. PITCHER VS OPPOSING TEAM
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "BUILDING PITCHER MATCHUP DATABASE"
)

print(
    "=" * 80
)


pitcher_group = (
    df
    .groupby(
        [
            "pitcher",
            "opposing_team"
        ]
    )
    .agg(
        plate_appearances=(
            "batter",
            "size"
        ),

        hits_allowed=(
            "is_hit",
            "sum"
        ),

        home_runs_allowed=(
            "is_hr",
            "sum"
        ),

        walks_allowed=(
            "is_walk",
            "sum"
        ),

        strikeouts=(
            "is_strikeout",
            "sum"
        ),

        xBA_allowed=(
            "estimated_ba_using_speedangle",
            "mean"
        ),

        xwOBA_allowed=(
            "estimated_woba_using_speedangle",
            "mean"
        ),

        exit_velocity_allowed=(
            "launch_speed",
            "mean"
        ),

        hard_hit_rate_allowed=(
            "hard_hit",
            "mean"
        )
    )
    .reset_index()
)


pitcher_group[
    "hit_rate_allowed"
] = np.where(
    pitcher_group[
        "plate_appearances"
    ] > 0,

    pitcher_group[
        "hits_allowed"
    ]
    /
    pitcher_group[
        "plate_appearances"
    ],

    np.nan
)


pitcher_group[
    "strikeout_rate"
] = np.where(
    pitcher_group[
        "plate_appearances"
    ] > 0,

    pitcher_group[
        "strikeouts"
    ]
    /
    pitcher_group[
        "plate_appearances"
    ],

    np.nan
)


pitcher_group[
    "walk_rate"
] = np.where(
    pitcher_group[
        "plate_appearances"
    ] > 0,

    pitcher_group[
        "walks_allowed"
    ]
    /
    pitcher_group[
        "plate_appearances"
    ],

    np.nan
)


joblib.dump(
    pitcher_group,
    os.path.join(
        ROOT,
        "pitcher_matchups.joblib"
    )
)


print(
    f"Saved pitcher_matchups.joblib "
    f"({len(pitcher_group):,} pitcher/team records)"
)


# ================================================================
# 4. BATTER VS PITCHER
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "BUILDING BATTER VS PITCHER DATABASE"
)

print(
    "=" * 80
)


batter_pitcher = (
    df
    .groupby(
        [
            "batter",
            "pitcher"
        ]
    )
    .agg(
        plate_appearances=(
            "game_pk",
            "size"
        ),

        hits=(
            "is_hit",
            "sum"
        ),

        home_runs=(
            "is_hr",
            "sum"
        ),

        walks=(
            "is_walk",
            "sum"
        ),

        strikeouts=(
            "is_strikeout",
            "sum"
        ),

        xBA=(
            "estimated_ba_using_speedangle",
            "mean"
        ),

        xwOBA=(
            "estimated_woba_using_speedangle",
            "mean"
        ),

        exit_velocity=(
            "launch_speed",
            "mean"
        ),

        hard_hit_rate=(
            "hard_hit",
            "mean"
        )
    )
    .reset_index()
)


batter_pitcher[
    "hit_rate"
] = np.where(
    batter_pitcher[
        "plate_appearances"
    ] > 0,

    batter_pitcher[
        "hits"
    ]
    /
    batter_pitcher[
        "plate_appearances"
    ],

    np.nan
)


batter_pitcher[
    "strikeout_rate"
] = np.where(
    batter_pitcher[
        "plate_appearances"
    ] > 0,

    batter_pitcher[
        "strikeouts"
    ]
    /
    batter_pitcher[
        "plate_appearances"
    ],

    np.nan
)


batter_pitcher[
    "walk_rate"
] = np.where(
    batter_pitcher[
        "plate_appearances"
    ] > 0,

    batter_pitcher[
        "walks"
    ]
    /
    batter_pitcher[
        "plate_appearances"
    ],

    np.nan
)


joblib.dump(
    batter_pitcher,
    os.path.join(
        ROOT,
        "batter_matchups.joblib"
    )
)


print(
    f"Saved batter_matchups.joblib "
    f"({len(batter_pitcher):,} batter/pitcher records)"
)


# ================================================================
# 5. RECENT TEAM FORM
#
# Fast version.
#
# Rather than rebuilding the entire Last 5/10 Statcast pipeline,
# create compact recent windows from the existing PA data.
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "BUILDING FAST RECENT TEAM FORM"
)

print(
    "=" * 80
)


# Determine the most recent completed games
# represented in the parquet.

recent_games = (
    games
    .sort_values(
        "game_date"
    )
    .tail(20)
    .copy()
)


recent_df = df[
    df["game_pk"].isin(
        recent_games[
            "game_pk"
        ]
    )
].copy()


recent_rows = []


for team, group in recent_df.groupby(
    "batting_team"
):

    games_played = (
        group[
            "game_pk"
        ]
        .nunique()
    )

    pa = len(group)

    hits = int(
        group["is_hit"].sum()
    )

    hrs = int(
        group["is_hr"].sum()
    )

    walks = int(
        group["is_walk"].sum()
    )

    strikeouts = int(
        group["is_strikeout"].sum()
    )

    recent_rows.append(
        {
            "team":
                team,

            "games":
                games_played,

            "plate_appearances":
                pa,

            "hits":
                hits,

            "home_runs":
                hrs,

            "walks":
                walks,

            "strikeouts":
                strikeouts,

            "hit_rate":
                hits / pa
                if pa
                else np.nan,

            "walk_rate":
                walks / pa
                if pa
                else np.nan,

            "strikeout_rate":
                strikeouts / pa
                if pa
                else np.nan,

            "xBA":
                group[
                    "estimated_ba_using_speedangle"
                ].mean(),

            "xwOBA":
                group[
                    "estimated_woba_using_speedangle"
                ].mean(),

            "exit_velocity":
                group[
                    "launch_speed"
                ].mean(),

            "hard_hit_rate":
                group[
                    "hard_hit"
                ].mean()
        }
    )


recent_team_stats = pd.DataFrame(
    recent_rows
)


joblib.dump(
    recent_team_stats,
    os.path.join(
        ROOT,
        "team_recent_stats.joblib"
    )
)


print(
    f"Saved team_recent_stats.joblib "
    f"({len(recent_team_stats)} teams)"
)


# ================================================================
# FINISH
# ================================================================

print(
    "\n" +
    "=" * 80
)

print(
    "CACHE BUILD COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nCreated:"
)

print(
    "  [OK] team_season_stats.joblib"
)

print(
    "  [OK] h2h_stats.joblib"
)

print(
    "  [OK] pitcher_matchups.joblib"
)

print(
    "  [OK] batter_matchups.joblib"
)

print(
    "  [OK] team_recent_stats.joblib"
)

print(
    "\nPrediction engine now has access to:"
)

print(
    "  | Season-long team offence"
)

print(
    "  | Recent team form"
)

print(
    "  | Historical team H2H"
)

print(
    "  | Pitcher vs opposing team history"
)

print(
    "  | Batter vs opposing pitcher history"
)

print(
    "  | Statcast xBA"
)

print(
    "  | Statcast xwOBA"
)

print(
    "  | Exit velocity"
)

print(
    "  | Hard-hit rate"
)

print(
    "\nYou do NOT need to rebuild these caches "
    "every time you run predictions."
)

print(
    "Rebuild them when you want to refresh the "
    "underlying historical/recent data."
)

print()