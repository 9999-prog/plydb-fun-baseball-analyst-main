"""Pregame MLB matchup and prop analytics.

The predictor combines trained model artifacts, time-safe historical Statcast,
current-season MLB Stats API context, and optional market prices. Every
probability is accompanied by evidence and freshness metadata; missing or
stale data is not converted into a confident value claim.

Run:
    python predict_todays_games.py
    python predict_todays_games.py 2026-08-15
    python predict_todays_games.py 2026-08-15 --speak
"""

import os
import sys
import math
import json
import time
import requests
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib

# Keep the rich box-drawing output usable on Windows consoles configured for a
# legacy code page.  Replacing an unencodable glyph is better than aborting a
# full slate after all model work has completed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from prop_metrics import (
    projected_score_profile,
    projected_nrfi_prob,
    projected_rifi_prob,
    totals_pick,
    best_total_market,
    fair_american_odds,
    prop_reason_under,
    prop_reason_nrfi,
)
from modern_stats import fetch_modern_team_stats, fetch_modern_player_stats
from terminal_theme import (
    bold as terminal_bold,
    configure_terminal_display,
    dim as terminal_dim,
    purple as terminal_purple,
    white as terminal_white,
)

from advanced_metrics import (
    build_batter_metric_card,
    build_team_metric_cards,
    confidence_adjusted_probability,
    json_safe,
    league_reference_context,
    market_card,
    metric_definitions,
    team_h2h_metrics,
    team_name_matches,
)
from predictor_utils import clean_sentence, pick_annotation, safe_error

# NEW: Config, logging, decimal odds
from config_loader import (
    get_config, get_odds_config, get_win_model_weights, get_batter_model_weights,
    get_shrinkage_params, get_validation_params, get_data_freshness_params,
    get_feature_params, get_paths, get_timezone
)
from logging_utils import setup_logging, get_logger, PredictionLogger, LogTimer
from odds_decimal import (
    decimal_to_probability, probability_to_decimal, expected_value_decimal,
    kelly_fraction, no_vig_probabilities, best_decimal_odds,
    american_to_decimal, decimal_to_american
)

warnings.filterwarnings("ignore")

# The analyst's visible terminal presentation is intentionally limited to
# white and purple. This also makes a best-effort 1.4x Windows console font
# adjustment when the process is attached to a compatible interactive host.
configure_terminal_display()

# Setup structured logging
logger = setup_logging("mlb_analyst.predictor")
pred_logger = PredictionLogger(logger)


# ================================================================
# LOAD CONFIGURATION
# ================================================================

config = get_config()
odds_config = get_odds_config()
win_weights = get_win_model_weights()
batter_weights = get_batter_model_weights()
shrinkage = get_shrinkage_params()
validation = get_validation_params()
data_freshness = get_data_freshness_params()
features_config = get_feature_params()
paths_config = get_paths()
TIMEZONE_STR = get_timezone()

ODDS_API_KEY = odds_config.get("api_key") or os.getenv("ODDS_API_KEY", "").strip()

BASEBALL_TZ = ZoneInfo(TIMEZONE_STR)
LOCAL_TZ = datetime.now().astimezone().tzinfo or BASEBALL_TZ

# Load constants from config
H2H_LOOKBACK = features_config.get("h2h_lookback", 10)
H2H_MIN_GAMES = validation.get("min_h2h_games", 3)
BVP_MIN_PA = validation.get("min_bvp_pa", 8)
LINEUP_GAMES = features_config.get("lineup_games", 20)
N_BATTERS = features_config.get("n_batters", 9)
BVP_SHRINK_K = shrinkage.get("bvp_shrink_k", 12)
H2H_SHRINK_K = shrinkage.get("h2h_shrink_k", 6)
RECENCY_HALF_LIFE_GAMES = shrinkage.get("recency_half_life_games", 4.0)

# Win model weights from config
W_MODEL = win_weights.get("model", 0.42)
W_PYTHAG = win_weights.get("pythag", 0.14)
W_RECENT_FORM = win_weights.get("recent_form", 0.18)
W_SP_QUALITY = win_weights.get("sp_quality", 0.10)
W_BULLPEN = win_weights.get("bullpen", 0.08)
W_LINEUP = win_weights.get("lineup", 0.08)
REST_ADVANTAGE_MAX = win_weights.get("rest_advantage_max", 0.02)
H2H_NUDGE_MAX = win_weights.get("h2h_nudge_max", 0.02)

# Batter model weights from config
WB_MODEL = batter_weights.get("model", 0.40)
WB_RECENT_FORM = batter_weights.get("recent_form", 0.25)
WB_PITCHER_ALLOWED = batter_weights.get("pitcher_allowed", 0.15)
WB_PLATOON = batter_weights.get("platoon", 0.10)
WB_PARK = batter_weights.get("park", 0.08)
WB_BVP = batter_weights.get("bvp", 0.02)

# Validation thresholds
MIN_EDGE_WIN = validation.get("min_edge_win", 0.04)
MIN_EDGE_BATTER = validation.get("min_edge_batter", 0.03)
MIN_SIGNAL_QUALITY = validation.get("min_signal_quality", 0.50)
MAX_STALENESS_HIGH_CONF = validation.get("max_staleness_days_high_confidence", 14)
MAX_STALENESS_ANY_BET = validation.get("max_staleness_days_any_bet", 60)
CLI_ARGS = sys.argv[1:]
SPEAK_OUTPUT = (
    "--speak" in CLI_ARGS
    or os.getenv("PREDICTOR_SPEAK", "").strip().lower() in {"1", "true", "yes", "on"}
)
SPEAK_STYLE = os.getenv("PREDICTOR_SPEAK_STYLE", "sarcastic")
SPEAK_BACKEND = os.getenv("PREDICTOR_SPEAK_BACKEND", "auto")

def _cli_date(value):
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False

DATE = next(
    (arg for arg in CLI_ARGS if _cli_date(arg)),
    datetime.now(LOCAL_TZ).strftime("%Y-%m-%d"),
)

ROOT = os.path.dirname(os.path.abspath(__file__))
STATCAST_FILE = os.path.join(
    ROOT, "data", "pybaseball", "statcast",
    "statcast_multiseason_pa_level_model_ready.parquet"
)
# Raw Statcast file for staleness check (updated with recent 2026 data)
RAW_STATCAST_FILE = os.path.join(
    ROOT, "data", "pybaseball", "statcast",
    "statcast.parquet"
)


# ================================================================
# DISPLAY AND NUMERIC HELPERS
# ================================================================

def line():
    print(terminal_white("=" * 80))

def subline():
    print(terminal_white("-" * 80))

def safe_num(value, default=np.nan):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default

def valid(value):
    value = safe_num(value)
    return math.isfinite(value)

def clamp(value, low=0.0, high=1.0):
    value = safe_num(value)
    if not valid(value):
        return np.nan
    return max(low, min(high, value))

def pct(value):
    value = safe_num(value)
    if not valid(value):
        return "N/A"
    return term_purple(f"{value * 100:.1f}%")

def stat_pct(value, digits=1):
    value = safe_num(value)
    if not valid(value):
        return "N/A"
    return term_purple(f"{value * 100:.{digits}f}%")

def number(value, digits=3):
    value = safe_num(value)
    if not valid(value):
        return "N/A"
    return term_purple(f"{value:.{digits}f}")

def logit(p, eps=1e-4):
    p = clamp(p, eps, 1 - eps)
    if not valid(p):
        return np.nan
    return math.log(p / (1 - p))

def inv_logit(x):
    if not valid(x):
        return np.nan
    return 1.0 / (1.0 + math.exp(-x))


def term_bold(text):
    return terminal_bold(text)


def term_dim(text):
    return terminal_dim(text)


def term_light(text):
    return terminal_white(text)


def term_purple(text):
    return terminal_purple(text)


def model_predict_proba(model, X_df):
    """Support sklearn and XGBoost-style models without crashing when one is used."""
    if model is None:
        return np.array([[0.5, 0.5]])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_df)
    if hasattr(model, "predict"):
        preds = model.predict(X_df)
        if preds.ndim == 1:
            probs = np.zeros((len(preds), 2))
            for i, p in enumerate(preds):
                probs[i, 0] = 1.0 - float(clamp(p, 0.0, 1.0))
                probs[i, 1] = float(clamp(p, 0.0, 1.0))
            return probs
        return preds
    raise TypeError(f"Model type {type(model)} has no predict_proba or predict method.")


def signal_quality_score(
    game,
    form_home,
    form_away,
    sp_home,
    sp_away,
    bp_home,
    bp_away,
    pyth_home,
    pyth_away,
    *,
    model_available=False,
    modern_games_home=0,
    modern_games_away=0,
    n_home_games=0,
    n_away_games=0,
    n_bp_home=0,
    n_bp_away=0,
    pyth_available_home=False,
    pyth_available_away=False,
    local_staleness_days=None,
):
    """Score evidence availability, not apparent disagreement between priors."""
    score = 0.0
    if model_available:
        score += 0.25
    if game.get("home_pitcher_id") is not None and game.get("away_pitcher_id") is not None:
        score += 0.20
    if min(int(modern_games_home or 0), int(modern_games_away or 0)) >= 5:
        score += 0.15
    if min(int(n_home_games or 0), int(n_away_games or 0)) >= 5:
        score += 0.10
    if min(int(n_bp_home or 0), int(n_bp_away or 0)) >= 5:
        score += 0.10
    if pyth_available_home and pyth_available_away:
        score += 0.10
    if local_staleness_days is None or safe_num(local_staleness_days, 9999) <= 14:
        score += 0.10
    elif safe_num(local_staleness_days, 9999) <= 60:
        score += 0.05
    return clamp(score, 0.0, 1.0)


def shrink(sample_rate, sample_n, prior_rate, prior_k):
    """Bayesian-style shrinkage: small samples pulled toward the prior."""
    sample_rate = safe_num(sample_rate)
    prior_rate = safe_num(prior_rate, default=0.5)
    sample_n = safe_num(sample_n, default=0)
    if not valid(sample_rate) or sample_n <= 0:
        return prior_rate
    return ((sample_rate * sample_n) + (prior_rate * prior_k)) / (sample_n + prior_k)

def recency_weights(n, half_life=RECENCY_HALF_LIFE_GAMES):
    """Most-recent-game-first exponential decay weights, sum to 1."""
    if n <= 0:
        return np.array([])
    decay = math.log(2) / half_life
    idx = np.arange(n)  # 0 = most recent
    w = np.exp(-decay * idx)
    return w / w.sum()


def load_file(filename):
    path = os.path.join(ROOT, filename)
    if not os.path.exists(path):
        print(f"WARNING: {filename} not found.")
        return None
    try:
        data = joblib.load(path)
        print(f"Loaded {filename}")
        return data
    except Exception as exc:
        print(f"WARNING loading {filename}: {safe_error(exc, secrets=[ODDS_API_KEY])}")
        return None


def effective_as_of_date(as_of_date=None):
    """Cap historical windows without dropping the latest completed data day."""
    target = pd.Timestamp(as_of_date or DATE)
    if pa_df is None or pa_df.empty or "game_date" not in pa_df.columns:
        return target
    latest = pd.to_datetime(pa_df["game_date"], errors="coerce").max()
    if pd.isna(latest):
        return target
    # Historical helpers use an exclusive upper bound.  Add one calendar day
    # when the requested forecast is after the local data so the latest
    # completed game day remains usable, while same-day predictions still
    # exclude outcomes from that day.
    return min(target, latest + timedelta(days=1))


# ================================================================
# HEADER
# ================================================================

print()
line()
print("MLB ANALYTICS PREDICTOR")
line()
print(f"Prediction date: {DATE}")
print()


# ================================================================
# LOAD MODELS AND CACHES
# ================================================================

hit_bundle = load_file("hit_model.joblib")
win_bundle = load_file("win_model.joblib")
snapshots = load_file("current_form_snapshots.joblib")
team_season = load_file("team_season_stats.joblib")
team_recent = load_file("team_recent_stats.joblib")
h2h_cache = load_file("h2h_stats.joblib")
pitcher_cache = load_file("pitcher_matchups.joblib")
batter_cache = load_file("batter_matchups.joblib")

if hit_bundle is None:
    raise RuntimeError("hit_model.joblib is required.")
if win_bundle is None:
    raise RuntimeError("win_model.joblib is required.")
if snapshots is None:
    raise RuntimeError("current_form_snapshots.joblib is required.")

hit_model = hit_bundle["model"]
HIT_FEATURES = hit_bundle["features"]
win_model = win_bundle["model"]
WIN_FEATURES = win_bundle["features"]

batter_snap = snapshots["batter_snapshot"]
pitcher_snap = snapshots["pitcher_snapshot"]
platoon_snap = snapshots["platoon_snapshot"]
park_snap = snapshots["park_snapshot"]

# Candidate scoring can touch hundreds of roster players. Build narrow lookup
# tables once instead of scanning the full historical cache for every player.
def _keyed_rows(frame, key_columns, value_columns=None):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    columns = list(dict.fromkeys(c for c in key_columns + (value_columns or list(frame.columns)) if c in frame.columns))
    work = frame[columns].copy()
    for column in key_columns:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    output = {}
    for values in work.itertuples(index=False, name=None):
        record = dict(zip(columns, values))
        key_values = []
        for column in key_columns:
            value = record.get(column)
            if column in {"batter", "pitcher"}:
                value = safe_num(value)
                value = int(value) if valid(value) else None
            else:
                value = str(value).upper() if value is not None and not pd.isna(value) else ""
            key_values.append(value)
        if all(value is not None for value in key_values):
            output[tuple(key_values) if len(key_values) > 1 else key_values[0]] = record
    return output


BATTER_SNAPSHOT_LOOKUP = _keyed_rows(
    batter_snap, ["batter"], HIT_FEATURES
)
PITCHER_SNAPSHOT_LOOKUP = _keyed_rows(
    pitcher_snap, ["pitcher"]
)
PLATOON_LOOKUP = _keyed_rows(
    platoon_snap, ["batter", "p_throws"]
)
BVP_RATE_LOOKUP = _keyed_rows(
    batter_cache, ["batter", "pitcher"],
    ["hit_rate", "plate_appearances", "pa"],
)
PITCHER_MATCHUP_LOOKUP = _keyed_rows(
    pitcher_cache, ["pitcher", "opposing_team"]
)


# ================================================================
# LOAD STATCAST FOR BULLPEN AND RECENCY FEATURES
# ================================================================

if not os.path.exists(STATCAST_FILE):
    raise RuntimeError(f"Statcast file not found:\n{STATCAST_FILE}")

print("\nLoading Statcast...")
pa_df = pd.read_parquet(STATCAST_FILE)
print(f"Loaded {len(pa_df):,} plate appearances.")

# Load raw Statcast file for staleness check (has recent 2026 data)
raw_df = pd.read_parquet(RAW_STATCAST_FILE)
raw_df["game_date"] = pd.to_datetime(raw_df["game_date"], errors="coerce")
RAW_LATEST_DATE = raw_df["game_date"].max()
RAW_EARLIEST_DATE = raw_df["game_date"].min()
try:
    DATA_STALENESS_DAYS = max(0, (pd.Timestamp(DATE) - RAW_LATEST_DATE).days)
except Exception:
    DATA_STALENESS_DAYS = None
if pd.notna(RAW_LATEST_DATE):
    age_label = f"{DATA_STALENESS_DAYS} days old" if DATA_STALENESS_DAYS is not None else "age unavailable"
    print(
        f"Local Statcast coverage: {RAW_EARLIEST_DATE.date()} -> "
        f"{RAW_LATEST_DATE.date()} ({age_label})."
    )
    if DATA_STALENESS_DAYS is not None and DATA_STALENESS_DAYS > 14:
        print(
            "WARNING: local player-level Statcast is stale; current-season team "
            "API context will be shown separately and old player signals should "
            "not be treated as live form."
        )
    # Additional warning for severe staleness
    if DATA_STALENESS_DAYS is not None and DATA_STALENESS_DAYS > 30:
        print(
            "CRITICAL WARNING: Local Statcast data is severely stale (>30 days). "
            "Prediction reliability may be significantly reduced."
        )

for column in ["home_team", "away_team", "inning_topbot", "p_throws", "stand"]:
    if column in pa_df.columns:
        pa_df[column] = pa_df[column].fillna("").astype(str)

for column in ["game_pk", "batter", "pitcher"]:
    if column in pa_df.columns:
        pa_df[column] = pd.to_numeric(pa_df[column], errors="coerce")

pa_df["batting_team"] = np.where(
    pa_df["inning_topbot"].str.lower().eq("bot"),
    pa_df["home_team"], pa_df["away_team"]
)
# the team whose PITCHER is on the mound for this PA
pa_df["pitching_team"] = np.where(
    pa_df["inning_topbot"].str.lower().eq("bot"),
    pa_df["away_team"], pa_df["home_team"]
)

events = pa_df["events"].fillna("").astype(str).str.lower()
pa_df["is_hit"] = events.isin({"single", "double", "triple", "home_run"}).astype(int)
pa_df["is_hr"] = events.eq("home_run").astype(int)
pa_df["is_walk"] = events.isin({"walk", "intent_walk"}).astype(int)
pa_df["is_strikeout"] = events.eq("strikeout").astype(int)

# runs scored on this PA, if the column exists (bat_score change is the
# standard Statcast way - fall back gracefully if not present)
if "bat_score" in pa_df.columns and "post_bat_score" in pa_df.columns:
    pa_df["runs_on_pa"] = (
        pd.to_numeric(pa_df["post_bat_score"], errors="coerce")
        - pd.to_numeric(pa_df["bat_score"], errors="coerce")
    ).clip(lower=0).fillna(0)
else:
    pa_df["runs_on_pa"] = 0.0

for column in ["launch_speed", "estimated_ba_using_speedangle",
               "estimated_woba_using_speedangle"]:
    if column not in pa_df.columns:
        pa_df[column] = np.nan
    pa_df[column] = pd.to_numeric(pa_df[column], errors="coerce")

pa_df["hard_hit"] = pa_df["launch_speed"].ge(95).fillna(False).astype(int)

# who started each game for each team (first pitcher to appear, by inning/pa
# order) - used to separate "starter" from "bullpen" workload
pa_df_sorted = pa_df.sort_values(
    ["game_pk", "game_date", "at_bat_number"],
    kind="mergesort",
)
starters = (
    pa_df_sorted.dropna(subset=["game_pk", "pitcher"])
    .groupby(["game_pk", "pitching_team"])["pitcher"]
    .first()
    .rename("starter_id")
    .reset_index()
)
pa_df = pa_df.merge(starters, on=["game_pk", "pitching_team"], how="left")
pa_df["is_bullpen_pa"] = (pa_df["pitcher"] != pa_df["starter_id"]).astype(int)


# ================================================================
# BULLPEN QUALITY  (last ~15 team appearances, from your own data)
# ================================================================

def team_bullpen_quality(team, as_of_date, lookback_games=15):
    """
    Runs allowed per PA by non-starting pitchers, recency-weighted,
    over the team's last `lookback_games`. Lower = better bullpen.
    Returns a 0-1 "bullpen strength" score (higher = better) so it can be
    blended the same direction as the other signals.
    """
    effective_date = effective_as_of_date(as_of_date)
    sub = pa_df[
        (pa_df["pitching_team"] == team)
        & (pa_df["is_bullpen_pa"] == 1)
        & (pa_df["game_date"] < pd.Timestamp(effective_date))
    ].copy()

    if sub.empty:
        return 0.5, 0  # neutral prior, no sample

    game_order = (
        sub[["game_pk", "game_date"]]
        .drop_duplicates()
        .sort_values("game_date", ascending=False)
        .head(lookback_games)
    )
    sub = sub[sub["game_pk"].isin(game_order["game_pk"])]

    per_game = (
        sub.groupby("game_pk")
        .agg(runs=("runs_on_pa", "sum"), pa=("runs_on_pa", "size"))
        .join(game_order.set_index("game_pk"))
        .sort_values("game_date", ascending=False)
    )
    n = len(per_game)
    if n == 0:
        return 0.5, 0

    w = recency_weights(n)
    runs_per_pa = (per_game["runs"] / per_game["pa"].clip(lower=1)).values
    weighted_rpa = float(np.dot(w, runs_per_pa))

    # league-average bullpen runs/PA is roughly ~0.115-0.13; convert to a
    # 0-1 "strength" score centered at 0.5 (lower runs allowed = higher score)
    league_avg = 0.12
    strength = clamp(0.5 - (weighted_rpa - league_avg) * 4.0, 0.05, 0.95)
    return strength, n


# ================================================================
# RECENCY-WEIGHTED TEAM FORM  (replaces flat last-5/last-10 avg)
# ================================================================

def team_recent_form(team, as_of_date, lookback_games=10):
    """
    Recency-weighted run differential per game over the team's last
    `lookback_games`, converted to a 0-1 "form strength" score.
    """
    effective_date = effective_as_of_date(as_of_date)
    bat = pa_df[
        (pa_df["batting_team"] == team)
        & (pa_df["game_date"] < pd.Timestamp(effective_date))
    ]
    pit = pa_df[
        (pa_df["pitching_team"] == team)
        & (pa_df["game_date"] < pd.Timestamp(effective_date))
    ]
    if bat.empty or pit.empty:
        return 0.5, 0

    games = (
        bat[["game_pk", "game_date"]]
        .drop_duplicates()
        .sort_values("game_date", ascending=False)
        .head(lookback_games)
    )
    if games.empty:
        return 0.5, 0

    scored = bat[bat["game_pk"].isin(games["game_pk"])].groupby("game_pk")["runs_on_pa"].sum()
    allowed = pit[pit["game_pk"].isin(games["game_pk"])].groupby("game_pk")["runs_on_pa"].sum()

    ordered = games.set_index("game_pk").join(scored.rename("scored")).join(allowed.rename("allowed"))
    ordered = ordered.sort_values("game_date", ascending=False).fillna(0)

    n = len(ordered)
    w = recency_weights(n)
    run_diff = (ordered["scored"] - ordered["allowed"]).values
    weighted_diff = float(np.dot(w, run_diff))

    # squash run differential per game into 0-1 (roughly +/-4 runs -> extremes)
    strength = clamp(0.5 + weighted_diff / 8.0, 0.05, 0.95)
    return strength, n


# ================================================================
# PYTHAGOREAN WIN EXPECTATION  (season R/RA via MLB Stats API)
# ================================================================

def fetch_team_season_run_stats(season_year):
    """
    Pulls season runs scored / runs allowed for every team from the
    public MLB Stats API team-stats endpoint. Used for Pythagorean
    win expectation, which published research shows is the single
    strongest individual predictor of MLB outcomes.
    """
    out = {}
    try:
        url = "https://statsapi.mlb.com/api/v1/teams/stats"
        resp = requests.get(
            url,
            params={
                "stats": "season",
                "group": "hitting",
                "season": int(season_year),
                "sportIds": 1,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        for split in data.get("stats", []):
            for s in split.get("splits", []):
                team_abbr = s.get("team", {}).get("abbreviation")
                runs = safe_num(s.get("stat", {}).get("runs"))
                if team_abbr and valid(runs):
                    out.setdefault(team_abbr, {})["runs_for"] = runs

        url2 = "https://statsapi.mlb.com/api/v1/teams/stats"
        resp2 = requests.get(
            url2,
            params={
                "stats": "season",
                "group": "pitching",
                "season": int(season_year),
                "sportIds": 1,
            },
            timeout=20,
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        for split in data2.get("stats", []):
            for s in split.get("splits", []):
                team_abbr = s.get("team", {}).get("abbreviation")
                runs_allowed = safe_num(s.get("stat", {}).get("runs"))
                era = safe_num(s.get("stat", {}).get("era"))
                whip = safe_num(s.get("stat", {}).get("whip"))
                if team_abbr:
                    out.setdefault(team_abbr, {})["runs_against"] = runs_allowed
                    out.setdefault(team_abbr, {})["era"] = era
                    out.setdefault(team_abbr, {})["whip"] = whip
    except Exception as exc:
        print(f"WARNING: could not fetch season run stats: {safe_error(exc)}")
    return out


def pythagorean_win_pct(runs_for, runs_against, exponent=1.83):
    runs_for = safe_num(runs_for)
    runs_against = safe_num(runs_against)
    if not valid(runs_for) or not valid(runs_against) or runs_for + runs_against <= 0:
        return 0.5
    rf = max(runs_for, 1e-6) ** exponent
    ra = max(runs_against, 1e-6) ** exponent
    return rf / (rf + ra)


season_year = pd.Timestamp(DATE).year
print(f"\nLoading {season_year} season run stats (Pythagorean expectation)...")
season_run_stats = fetch_team_season_run_stats(season_year)
modern_team_stats = fetch_modern_team_stats(season_year)
modern_player_stats = {}
print(f"Modern MLB Stats API team context: {len(modern_team_stats):,} teams")


# ================================================================
# STARTING PITCHER QUALITY  (ERA/FIP-proxy diff, from your pitcher_snap)
# ================================================================

def starter_quality_score(pitcher_id):
    """
    0-1 score, higher = better starter, built from your existing
    pitcher_snap (xBA/xwOBA/K%/BB% allowed). Falls back to a neutral
    0.5 if the pitcher isn't in the snapshot (e.g. rookie call-up, TBD).
    """
    if pitcher_id is None or pitcher_snap is None:
        return 0.5
    row = pitcher_snap.get(pitcher_id) if isinstance(pitcher_snap, dict) else None
    if row is None:
        try:
            row = pitcher_snap.loc[pitcher_id].to_dict()
        except Exception:
            return 0.5
    k_pct = safe_num(row.get("k_pct"))
    bb_pct = safe_num(row.get("bb_pct"))
    xwoba = safe_num(row.get("xwoba_against"))

    score = 0.5
    if valid(k_pct):
        score += (k_pct - 0.22) * 0.8       # league-avg K% roughly 22%
    if valid(bb_pct):
        score -= (bb_pct - 0.08) * 1.0      # league-avg BB% roughly 8%
    if valid(xwoba):
        score -= (xwoba - 0.32) * 1.2       # league-avg xwOBA roughly .320
    return clamp(score, 0.05, 0.95)


# ================================================================
# ODDS API
# ================================================================

def load_odds():
    if not ODDS_API_KEY:
        print("\nOdds API key not configured. Market prices and value edges are unavailable.")
        return []
    print("\nLoading MLB market odds...")
    odds_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "regions": "au",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    try:
        result = requests.get(odds_url, params=params, timeout=20)
        result.raise_for_status()
        print("Odds API loaded successfully.")
        return result.json()
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {401, 403}:
            print(f"WARNING: Odds API rejected credentials (HTTP {int(status)}); market data omitted.")
        else:
            print(f"WARNING: Odds API request failed ({safe_error(exc, secrets=[ODDS_API_KEY])}); market data omitted.")
        return []
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"WARNING: Odds API unavailable ({safe_error(exc, secrets=[ODDS_API_KEY])}); market data omitted.")
        return []

odds_data = load_odds()


def american_to_probability(odds):
    odds = safe_num(odds)
    if not valid(odds):
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    odds = safe_num(odds)
    if not valid(odds):
        return np.nan
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)

def best_team_price(team, opponent):
    """Return the robust best price for ``team`` in this matchup.

    Keep the printed price consistent with the metric cards: the helper uses
    the same no-vig consensus and outlier filter rather than selecting a
    lone malformed bookmaker quote.
    """
    market = market_card(odds_data, team, opponent)
    return market.get("home_best_price")


# ================================================================
# MLB SCHEDULE AND PROBABLE PITCHERS
# ================================================================

print("\nLoading MLB schedule and probable pitchers...")
schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
try:
    response = requests.get(
        schedule_url,
        params={
            "sportId": 1,
            "date": DATE,
            "hydrate": "probablePitcher,team",
        },
        timeout=20,
    )
    response.raise_for_status()
    schedule = response.json()
except Exception as exc:
    print(f"WARNING: MLB schedule unavailable for {DATE} ({safe_error(exc)}); slate is empty.")
    schedule = {"dates": []}

TEAM_ID_CACHE = {}
TEAM_LIST_CACHE = {}
ROSTER_CACHE = {}
PLAYER_NAME_CACHE = {}
RECENT_BATTER_IDS_CACHE = {}
SCORE_BATTER_CACHE = {}
BATTER_RECENT_FORM_CACHE = {}


def fetch_team_id(team_abbr, season_year=None):
    team_abbr = str(team_abbr).upper()
    season_year = int(season_year or pd.Timestamp(DATE).year)
    cache_key = (team_abbr, season_year)
    if cache_key in TEAM_ID_CACHE:
        return TEAM_ID_CACHE[cache_key]
    if season_year not in TEAM_LIST_CACHE:
        team_map = {}
        try:
            teams_url = "https://statsapi.mlb.com/api/v1/teams"
            teams_resp = requests.get(
                teams_url,
                params={"sportId": 1, "season": season_year},
                timeout=20,
            )
            teams_resp.raise_for_status()
            for team in teams_resp.json().get("teams", []):
                abbr = str(team.get("abbreviation", "")).upper()
                if abbr and team.get("id") is not None:
                    team_map[abbr] = int(team["id"])
                    # The current API occasionally uses AZ/ATH while older
                    # caches use ARI/OAK; keep the aliases in one map too.
                    if abbr == "AZ":
                        team_map["ARI"] = int(team["id"])
                    if abbr == "ATH":
                        team_map["OAK"] = int(team["id"])
        except Exception:
            team_map = {}
        TEAM_LIST_CACHE[season_year] = team_map
    TEAM_ID_CACHE[cache_key] = TEAM_LIST_CACHE[season_year].get(team_abbr)
    return TEAM_ID_CACHE[cache_key]


def fetch_team_roster(team_abbr, season_year=None):
    team_abbr = str(team_abbr).upper()
    season_year = int(season_year or pd.Timestamp(DATE).year)
    cache_key = (team_abbr, season_year)
    if cache_key in ROSTER_CACHE:
        return ROSTER_CACHE[cache_key]
    team_id = fetch_team_id(team_abbr, season_year=season_year)
    if team_id is None:
        ROSTER_CACHE[cache_key] = []
        return []
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        resp = requests.get(
            url,
            params={"season": season_year, "rosterType": "40Man"},
            timeout=20,
        )
        if resp.status_code != 200:
            ROSTER_CACHE[cache_key] = []
            return []
        roster = resp.json().get("roster", [])
        players = []
        for item in roster:
            person = item.get("person") or {}
            pid = person.get("id")
            if pid is None:
                continue
            player_record = {
                "batter_id": int(pid),
                "name": person.get("fullName") or "Unknown",
                "bat_side": (person.get("batSide") or {}).get("code") or "R",
                "pitch_hand": (person.get("pitchHand") or {}).get("code") or "R",
            }
            players.append(player_record)
            PLAYER_NAME_CACHE[int(pid)] = player_record["name"]
        ROSTER_CACHE[cache_key] = players
        return players
    except Exception:  # noqa: BLE001
        ROSTER_CACHE[cache_key] = []
        return []


def player_handedness(player_id):
    try:
        resp = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=15)
        if resp.status_code == 200:
            person = (resp.json().get("people") or [{}])[0]
            bat_side = (person.get("batSide") or {}).get("code") or "R"
            return bat_side.upper()
    except Exception:
        pass
    return "R"


def recent_pitcher_for_team(team_abbr, recent_days=45):
    if pa_df is None or team_abbr is None:
        return None
    team_abbr = str(team_abbr).upper()
    ts = pd.Timestamp(DATE)
    sub = pa_df[
        (pa_df["pitching_team"].astype(str).str.upper() == team_abbr)
        & (pa_df["game_date"] >= ts - pd.Timedelta(days=recent_days))
        & (pa_df["game_date"] < ts + pd.Timedelta(days=1))
    ].copy()
    if sub.empty:
        return None
    counts = sub.groupby("pitcher").size().reset_index(name="pa_count")
    if counts.empty:
        return None
    top_pitcher = counts.sort_values("pa_count", ascending=False).iloc[0]["pitcher"]
    return int(pd.to_numeric(top_pitcher, errors="coerce")) if pd.notna(top_pitcher) else None


def recent_batter_ids_for_team(team_abbr, limit=8):
    if pa_df is None or team_abbr is None:
        return []
    team_abbr = str(team_abbr).upper()
    cache_key = (team_abbr, int(limit), str(DATE))
    if cache_key in RECENT_BATTER_IDS_CACHE:
        return list(RECENT_BATTER_IDS_CACHE[cache_key])
    ts = pd.Timestamp(DATE)
    sub = pa_df[
        (pa_df["batting_team"].astype(str).str.upper() == team_abbr)
        & (pa_df["game_date"] >= ts - pd.Timedelta(days=45))
        & (pa_df["game_date"] < ts + pd.Timedelta(days=1))
    ].copy()
    if sub.empty:
        roster = fetch_team_roster(team_abbr, season_year=int(pd.Timestamp(DATE).year))
        ids = [int(x["batter_id"]) for x in roster[:limit]]
        RECENT_BATTER_IDS_CACHE[cache_key] = ids
        return list(ids)

    agg = (
        sub.groupby("batter")
        .agg(pa=("batter", "size"), hits=("is_hit", "sum"))
        .reset_index()
    )
    if agg.empty:
        roster = fetch_team_roster(team_abbr, season_year=int(pd.Timestamp(DATE).year))
        ids = [int(x["batter_id"]) for x in roster[:limit]]
        RECENT_BATTER_IDS_CACHE[cache_key] = ids
        return list(ids)
    top = agg.sort_values(["pa", "hits"], ascending=[False, False]).head(limit)
    ids = [int(x) for x in pd.to_numeric(top["batter"], errors="coerce").dropna().tolist()]
    if len(ids) < limit:
        roster = fetch_team_roster(team_abbr, season_year=int(pd.Timestamp(DATE).year))
        roster_ids = [int(x["batter_id"]) for x in roster]
        for bid in roster_ids:
            if bid not in ids:
                ids.append(bid)
            if len(ids) >= limit:
                break
    ids = ids[:limit]
    RECENT_BATTER_IDS_CACHE[cache_key] = ids
    return list(ids)


def team_roster_batter_candidates(team, opponent_pitcher_id, as_of_date, limit=8):
    roster = fetch_team_roster(team, season_year=int(pd.Timestamp(as_of_date).year))
    if not roster:
        return []
    out = []
    seen = set()
    for player in roster[:20]:
        batter_id = int(player["batter_id"])
        if batter_id in seen:
            continue
        seen.add(batter_id)
        player_name = player.get("name") or resolve_batter_name(batter_id)
        p = score_batter(batter_id, player_name, team, opponent_pitcher_id, is_home=False)
        p["team"] = team
        p["opp_pitcher_id"] = opponent_pitcher_id
        p["batter_id"] = batter_id
        p["batter_name"] = player_name
        p["bat_side"] = player.get("bat_side") or player_handedness(batter_id)
        p["matchup_score"] = batter_matchup_score(p)
        p["upset_detected"] = p["final_prob"] > 0.33 and p["pitcher_xba_allowed"] > 0.31
        out.append(p)
    out = sorted(out, key=lambda x: x["matchup_score"], reverse=True)
    return out[:limit]


def pitcher_name_for_id(pitcher_id):
    if pitcher_id is None:
        return "TBD"
    try:
        roster_api = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
        resp = requests.get(roster_api, timeout=15)
        if resp.status_code == 200:
            person = resp.json().get("people", [{}])[0]
            full_name = person.get("fullName")
            if full_name:
                return full_name
    except Exception:
        pass
    return f"Pitcher {int(pitcher_id)}"


games = []
for date_entry in schedule.get("dates", []):
    for game in date_entry.get("games", []):
        home = game["teams"]["home"]
        away = game["teams"]["away"]
        home_pitcher = home.get("probablePitcher") or {}
        away_pitcher = away.get("probablePitcher") or {}
        home_team = home["team"]["abbreviation"]
        away_team = away["team"]["abbreviation"]
        home_pitcher_id = home_pitcher.get("id") or recent_pitcher_for_team(home_team)
        away_pitcher_id = away_pitcher.get("id") or recent_pitcher_for_team(away_team)
        games.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_pitcher_id": home_pitcher_id,
            "home_pitcher_name": home_pitcher.get("fullName") or pitcher_name_for_id(home_pitcher_id),
            "away_pitcher_id": away_pitcher_id,
            "away_pitcher_name": away_pitcher.get("fullName") or pitcher_name_for_id(away_pitcher_id),
        })

print(f"Found {len(games)} MLB games.")
if not games:
    sys.exit("No MLB games found.")

# Fetch current-season player aggregates only for the small set of likely
# lineup candidates.  This is real MLB Stats API context, cached by
# modern_stats.py; it is kept separate from the trained game-hit model because
# the API reports PA hit rate rather than the model's game-hit target.
modern_candidate_ids = set()
for scheduled_game in games:
    for scheduled_team in [scheduled_game["home_team"], scheduled_game["away_team"]]:
        modern_candidate_ids.update(
            recent_batter_ids_for_team(scheduled_team, limit=12)
        )
try:
    modern_player_stats = fetch_modern_player_stats(
        modern_candidate_ids, season=season_year
    )
    available_modern_players = sum(
        bool(row.get("available")) for row in modern_player_stats.values()
    )
    print(
        f"Modern MLB Stats API player context: {available_modern_players:,}/"
        f"{len(modern_player_stats):,} candidates"
    )
except Exception as exc:
    print(f"WARNING: current-season player context unavailable: {safe_error(exc)}")
    modern_player_stats = {}

print()
line()
print("TODAY'S PROBABLE PITCHERS")
line()
for game in games:
    print(f"{game['away_team']} @ {game['home_team']}: "
          f"{game['away_pitcher_name']} vs {game['home_pitcher_name']}")



# ================================================================
# COMPOSITE TEAM WIN SCORING
# ================================================================

def row_lookup(frame_or_dict, lookup_value, key_name="team"):
    """Return the first matching row from a DataFrame or dict-like cache."""
    if frame_or_dict is None:
        return {}

    if isinstance(frame_or_dict, dict):
        value = frame_or_dict.get(lookup_value)
        if isinstance(value, dict):
            return value
        if isinstance(value, pd.Series):
            return value.to_dict()
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return {}
            return value.iloc[0].to_dict()
        return {}

    if isinstance(frame_or_dict, pd.DataFrame):
        df = frame_or_dict.copy()
        if df.empty:
            return {}
        if key_name in df.columns:
            match = df[df[key_name].astype(str).str.upper() == str(lookup_value).upper()]
            if not match.empty:
                return match.iloc[0].to_dict()
        if "team" in df.columns:
            match = df[df["team"].astype(str).str.upper() == str(lookup_value).upper()]
            if not match.empty:
                return match.iloc[0].to_dict()
        if "pitcher" in df.columns and str(lookup_value).isdigit():
            match = df[pd.to_numeric(df["pitcher"], errors="coerce") == float(lookup_value)]
            if not match.empty:
                return match.iloc[0].to_dict()
        if "batter" in df.columns and str(lookup_value).isdigit():
            match = df[pd.to_numeric(df["batter"], errors="coerce") == float(lookup_value)]
            if not match.empty:
                return match.iloc[0].to_dict()
        return {}

    return {}


def team_row(feature_table, team):
    return row_lookup(feature_table, team, key_name="team")


def batter_name_for_id_fallback(batter_id):
    if batter_id is None:
        return "Unknown hitter"
    if isinstance(batter_snap, pd.DataFrame):
        try:
            df = batter_snap.copy()
            mask = pd.to_numeric(df.get("batter", pd.Series(index=df.index, dtype="float64")), errors="coerce") == safe_num(batter_id)
            if mask.any():
                row = df.loc[mask].iloc[0].to_dict()
                for key in ["batter_name", "player_name", "name", "full_name"]:
                    if key in row and str(row[key]).strip():
                        return str(row[key]).strip()
        except Exception:
            pass
    elif isinstance(batter_snap, dict):
        try:
            row = batter_snap.get(batter_id)
            if isinstance(row, pd.Series):
                row = row.to_dict()
            if isinstance(row, dict):
                for key in ["batter_name", "player_name", "name", "full_name"]:
                    if key in row and str(row[key]).strip():
                        return str(row[key]).strip()
        except Exception:
            pass
    return f"Batter {int(safe_num(batter_id, batter_id)) if valid(safe_num(batter_id)) else batter_id}"


def projected_lineup_strength(team, opponent_pitcher_id, as_of_date=None):
    """Average hitter strength in the recent projected lineup against the opposing starter."""
    if "score_batter" not in globals():
        return 0.5
    as_of_date = as_of_date or DATE
    if opponent_pitcher_id is None:
        return 0.5
    ids = recent_batter_ids_for_team(team, limit=12)
    if not ids:
        return 0.5
    scores = []
    for batter_id in ids:
        name = batter_name_for_id_fallback(batter_id)
        row = score_batter(batter_id, name, team, opponent_pitcher_id, is_home=False)
        scores.append(clamp(safe_num(row.get("final_prob"), 0.24), 0.05, 0.95))
    if not scores:
        return 0.5
    return float(np.mean(scores))


def pitcher_row(feature_table, pitcher_id):
    if pitcher_id is None:
        return {}
    return row_lookup(feature_table, pitcher_id, key_name="pitcher")


def batter_row(feature_table, batter_id):
    if batter_id is None:
        return {}
    return row_lookup(feature_table, batter_id, key_name="batter")


def feature_value(row, *names, default=np.nan):
    if not isinstance(row, dict):
        return default
    for name in names:
        if name in row:
            val = safe_num(row.get(name), default)
            if valid(val):
                return val
    return default


def h2h_row(frame_or_dict, team, opponent):
    if frame_or_dict is None:
        return {}

    if isinstance(frame_or_dict, dict):
        candidates = [
            frame_or_dict.get((team, opponent)),
            frame_or_dict.get(f"{team}_{opponent}"),
            frame_or_dict.get((opponent, team)),
            frame_or_dict.get(f"{opponent}_{team}"),
        ]
        for val in candidates:
            if isinstance(val, dict):
                return val
        return {}

    if isinstance(frame_or_dict, pd.DataFrame):
        df = frame_or_dict.copy()
        if df.empty:
            return {}

        team_col = "team" if "team" in df.columns else None
        opp_col = "opponent" if "opponent" in df.columns else None
        if team_col and opp_col:
            mask = (
                (df[team_col].astype(str).str.upper() == str(team).upper()) &
                (df[opp_col].astype(str).str.upper() == str(opponent).upper())
            )
            if mask.any():
                return df.loc[mask].iloc[0].to_dict()
            reverse = (
                (df[team_col].astype(str).str.upper() == str(opponent).upper()) &
                (df[opp_col].astype(str).str.upper() == str(team).upper())
            )
            if reverse.any():
                item = df.loc[reverse].iloc[0].to_dict()
                if "team" in item and "opponent" in item:
                    item["home_win_pct"] = item.get("home_win_pct")
                return item

    return {}


def score_matchup(game):
    home, away = game["home_team"], game["away_team"]

    team_home = team_row(team_season, home)
    team_away = team_row(team_season, away)
    home_recent = team_row(team_recent, home)
    away_recent = team_row(team_recent, away)

    home_pitcher_row = pitcher_row(pitcher_snap, game.get("home_pitcher_id"))
    away_pitcher_row = pitcher_row(pitcher_snap, game.get("away_pitcher_id"))

    row = {}
    row.update({
        "home_pitcher_roll_k_rate": feature_value(home_pitcher_row, "pitcher_roll_k_rate"),
        "home_pitcher_roll_bb_rate": feature_value(home_pitcher_row, "pitcher_roll_bb_rate"),
        "home_pitcher_roll_xba_against": feature_value(home_pitcher_row, "pitcher_roll_xba_against"),
        "home_pitcher_roll_hardhit_against": feature_value(home_pitcher_row, "pitcher_roll_hardhit_against"),
        "home_pitcher_roll_velo": feature_value(home_pitcher_row, "pitcher_roll_velo"),
        "home_pitcher_roll_spin": feature_value(home_pitcher_row, "pitcher_roll_spin"),
        "away_pitcher_roll_k_rate": feature_value(away_pitcher_row, "pitcher_roll_k_rate"),
        "away_pitcher_roll_bb_rate": feature_value(away_pitcher_row, "pitcher_roll_bb_rate"),
        "away_pitcher_roll_xba_against": feature_value(away_pitcher_row, "pitcher_roll_xba_against"),
        "away_pitcher_roll_hardhit_against": feature_value(away_pitcher_row, "pitcher_roll_hardhit_against"),
        "away_pitcher_roll_velo": feature_value(away_pitcher_row, "pitcher_roll_velo"),
        "away_pitcher_roll_spin": feature_value(away_pitcher_row, "pitcher_roll_spin"),
        "home_batting_hit_rate": feature_value(team_home, "hit_rate", "season_hit_rate"),
        "home_batting_xba": feature_value(team_home, "xBA", "xba", "season_xba"),
        "away_batting_hit_rate": feature_value(team_away, "hit_rate", "season_hit_rate"),
        "away_batting_xba": feature_value(team_away, "xBA", "xba", "season_xba"),
    })

    park_row = row_lookup(park_snap, home, key_name="home_team") if isinstance(park_snap, pd.DataFrame) else {}
    if isinstance(park_snap, dict):
        park_row = park_snap.get(home, {})
    row["park_hit_factor"] = feature_value(park_row, "park_hit_factor")
    row["park_hr_factor"] = feature_value(park_row, "park_hr_factor")

    X = pd.DataFrame([row], columns=WIN_FEATURES).reindex(columns=WIN_FEATURES)
    if not X.empty:
        for c in WIN_FEATURES:
            if c not in X.columns:
                X[c] = np.nan
        X = X[WIN_FEATURES].apply(pd.to_numeric, errors="coerce")

    model_available = False
    if team_season is not None and WIN_FEATURES and not X.empty and X.notna().all().all():
        try:
            model_prob_home = clamp(model_predict_proba(win_model, X)[0][1])
            model_available = valid(model_prob_home)
        except Exception:
            model_prob_home = np.nan
    else:
        model_prob_home = np.nan

    # Improved model availability handling: don't use model probability when unavailable
    if not model_available:
        print(f"WARNING: Win model unavailable for {home} vs {away}, falling back to priors only")
        model_prob_home = 0.5  # Neutral fallback when model unavailable
        # Reduce weight of model component when unavailable
        model_weight_adjustment = 0.0
    else:
        model_weight_adjustment = W_MODEL

    # 2) Pythagorean win expectation.  Prefer the modern adapter's current
    # season rates, with the existing endpoint as a non-synthetic fallback.
    hs = season_run_stats.get(home, {})
    as_ = season_run_stats.get(away, {})
    modern_home = modern_team_stats.get(str(home).upper(), {})
    modern_away = modern_team_stats.get(str(away).upper(), {})

    def modern_pythagorean(row, fallback):
        modern_for = safe_num(row.get("runs_for"))
        modern_against = safe_num(row.get("runs_allowed"))
        if valid(modern_for) and valid(modern_against):
            return pythagorean_win_pct(modern_for, modern_against)
        return fallback

    fallback_home = pythagorean_win_pct(hs.get("runs_for"), hs.get("runs_against"))
    fallback_away = pythagorean_win_pct(as_.get("runs_for"), as_.get("runs_against"))
    pyth_home = modern_pythagorean(modern_home, fallback_home)
    pyth_away = modern_pythagorean(modern_away, fallback_away)
    pyth_diff_score = clamp(0.5 + (pyth_home - pyth_away) / 2.0, 0.05, 0.95)

    # 3) recent form from real team-level rolling stats
    form_home, n_home_games = team_recent_form(home, DATE)
    form_away, n_away_games = team_recent_form(away, DATE)
    form_score = clamp(0.5 + (form_home - form_away) / 2.0, 0.05, 0.95)

    # 4) starting pitcher quality
    sp_home = starter_quality_score(game.get("home_pitcher_id"))
    sp_away = starter_quality_score(game.get("away_pitcher_id"))
    sp_score = clamp(0.5 + (sp_home - sp_away) / 2.0, 0.05, 0.95)

    # 5) bullpen quality
    bp_home, n_bp_home = team_bullpen_quality(home, DATE)
    bp_away, n_bp_away = team_bullpen_quality(away, DATE)
    bp_score = clamp(0.5 + (bp_home - bp_away) / 2.0, 0.05, 0.95)

    # 6) lineup strength against the opposing starter (projected contemporary edge)
    home_lineup = projected_lineup_strength(home, game.get("away_pitcher_id"), DATE)
    away_lineup = projected_lineup_strength(away, game.get("home_pitcher_id"), DATE)
    lineup_score = clamp(0.5 + (home_lineup - away_lineup) / 2.0, 0.05, 0.95)

    # Adjust weights when model is unavailable
    effective_W_MODEL = model_weight_adjustment if 'model_weight_adjustment' in locals() else W_MODEL
    if effective_W_MODEL == 0.0:
        # Renormalize weights when model is unavailable
        total_weight = W_PYTHAG + W_RECENT_FORM + W_SP_QUALITY + W_BULLPEN + W_LINEUP
        blended = (
            (W_PYTHAG / total_weight) * logit(pyth_diff_score)
            + (W_RECENT_FORM / total_weight) * logit(form_score)
            + (W_SP_QUALITY / total_weight) * logit(sp_score)
            + (W_BULLPEN / total_weight) * logit(bp_score)
            + (W_LINEUP / total_weight) * logit(lineup_score)
        )
    else:
        blended = (
            effective_W_MODEL * logit(model_prob_home)
            + W_PYTHAG * logit(pyth_diff_score)
            + W_RECENT_FORM * logit(form_score)
            + W_SP_QUALITY * logit(sp_score)
            + W_BULLPEN * logit(bp_score)
            + W_LINEUP * logit(lineup_score)
        )
    home_win_prob = inv_logit(blended)

    # Directional H2H is rebuilt from completed PA/game scores rather than
    # trusting a cache whose row orientation can be ambiguous.  The helper
    # shrinks small samples and refuses to create a signal when no real games
    # exist.
    h2h_signal = team_h2h_metrics(pa_df, home, away, DATE)
    h2h_l5 = h2h_signal.get("h2h_l5")
    h2h_l10 = h2h_signal.get("h2h_l10")
    
    # Validate H2H consistency: flag large discrepancies between l5 and l10
    if valid(h2h_l5) and valid(h2h_l10):
        h2h_diff = abs(h2h_l5 - h2h_l10)
        if h2h_diff > 0.2:  # More than 20% difference is suspicious
            print(f"WARNING: H2H inconsistency detected for {home} vs {away}: l5={h2h_l5:.3f}, l10={h2h_l10:.3f}")
    
    if (h2h_signal.get("h2h_games", 0) >= H2H_MIN_GAMES
            and valid(h2h_signal.get("h2h_delta"))):
        nudge = clamp(h2h_signal["h2h_delta"] * 2.0, -1.0, 1.0) * H2H_NUDGE_MAX
        home_win_prob = clamp(home_win_prob + nudge)

    quality = signal_quality_score(
        game,
        form_home,
        form_away,
        sp_home,
        sp_away,
        bp_home,
        bp_away,
        pyth_home,
        pyth_away,
        model_available=model_available,
        modern_games_home=modern_home.get("modern_games", 0),
        modern_games_away=modern_away.get("modern_games", 0),
        n_home_games=n_home_games,
        n_away_games=n_away_games,
        n_bp_home=n_bp_home,
        n_bp_away=n_bp_away,
        pyth_available_home=valid(modern_home.get("runs_for")) or valid(hs.get("runs_for")),
        pyth_available_away=valid(modern_away.get("runs_for")) or valid(as_.get("runs_for")),
        local_staleness_days=DATA_STALENESS_DAYS,
    )

    score_profile = projected_score_profile(
        home, away, pa_df, DATE, modern_stats=modern_team_stats
    )

    return {
        "home_team": home, "away_team": away,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
        "model_prob_home": model_prob_home,
        "model_available": model_available,
        "pyth_home": pyth_home, "pyth_away": pyth_away,
        "form_home": form_home, "form_away": form_away,
        "n_home_games": n_home_games, "n_away_games": n_away_games,
        "modern_games_home": int(modern_home.get("modern_games", 0) or 0),
        "modern_games_away": int(modern_away.get("modern_games", 0) or 0),
        "pyth_home_available": valid(modern_home.get("runs_for")) or valid(hs.get("runs_for")),
        "pyth_away_available": valid(modern_away.get("runs_for")) or valid(as_.get("runs_for")),
        "local_data_staleness_days": DATA_STALENESS_DAYS,
        "sp_home": sp_home, "sp_away": sp_away,
        "sp_home_available": bool(home_pitcher_row),
        "sp_away_available": bool(away_pitcher_row),
        "bp_home": bp_home, "bp_away": bp_away,
        "n_bp_home": n_bp_home, "n_bp_away": n_bp_away,
        "home_lineup_strength": home_lineup,
        "away_lineup_strength": away_lineup,
        "lineup_score": lineup_score,
        "home_pitcher_name": game["home_pitcher_name"],
        "away_pitcher_name": game["away_pitcher_name"],
        "h2h_l5": h2h_signal.get("h2h_l5"),
        "h2h_l10": h2h_signal.get("h2h_l10"),
        "h2h_delta": h2h_signal.get("h2h_delta"),
        "h2h_games": h2h_signal.get("h2h_games", 0),
        "h2h_status": h2h_signal.get("h2h_status", "NO_DATA"),
        "score_profile": score_profile,
        "signal_quality": quality,
    }


DEBUG_TIMING = os.getenv("PREDICTOR_DEBUG_TIMING", "").strip().lower() in {
    "1", "true", "yes", "on"
}

results = []
for game_index, scheduled_game in enumerate(games, start=1):
    started = time.perf_counter() if DEBUG_TIMING else None
    results.append(score_matchup(scheduled_game))
    if DEBUG_TIMING:
        print(
            f"Scored matchup {game_index}/{len(games)} in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )


# ================================================================
# MARKET EDGE
# ================================================================

def attach_edge(r):
    market = market_card(odds_data, r["home_team"], r["away_team"])
    home_price = market.get("home_best_price")
    away_price = market.get("away_best_price")
    r["home_market_prob"] = market.get("home_market_probability", np.nan)
    r["away_market_prob"] = market.get("away_market_probability", np.nan)
    r["home_edge"] = (
        r["home_win_prob"] - r["home_market_prob"]
        if valid(r.get("home_market_prob")) else np.nan
    )
    r["away_edge"] = (
        r["away_win_prob"] - r["away_market_prob"]
        if valid(r.get("away_market_prob")) else np.nan
    )
    r["home_price"] = home_price
    r["away_price"] = away_price
    r["market_book_count"] = market.get("book_count", 0)
    return r

results = [attach_edge(r) for r in results]


def attach_metric_cards(r):
    score_profile = r.get("score_profile") or projected_score_profile(
        r["home_team"], r["away_team"], pa_df, DATE,
        modern_stats=modern_team_stats,
    )
    h2h = {
        "h2h_l5": r.get("h2h_l5"),
        "h2h_l10": r.get("h2h_l10"),
        "h2h_delta": r.get("h2h_delta"),
        "h2h_games": r.get("h2h_games", 0),
        "h2h_status": r.get("h2h_status", "NO_DATA"),
    }
    r["score_profile"] = score_profile
    r["metric_cards"] = build_team_metric_cards(
        r,
        score_profile=score_profile,
        h2h=h2h,
        odds_data=odds_data,
        local_staleness_days=DATA_STALENESS_DAYS,
    )
    return r


results = [attach_metric_cards(r) for r in results]

# Totals/NRFI were previously recomputed once for the team section and again
# for the final totals section. On a stale local PA table that meant repeated
# full-frame scans and made a normal slate look hung. Cache one transparent
# bundle per matchup and reuse the score profile already built by score_matchup.
GAME_PROP_CACHE = {}


def game_prop_bundle(home_team, away_team, score_profile=None):
    key = (str(home_team).upper(), str(away_team).upper())
    if key not in GAME_PROP_CACHE:
        totals = totals_pick(
            home_team,
            away_team,
            pa_df,
            DATE,
            odds_data,
            modern_stats=modern_team_stats,
            score_profile=score_profile,
        )
        nrfi_probability = projected_nrfi_prob(
            home_team, away_team, pa_df, DATE
        )
        GAME_PROP_CACHE[key] = {
            "totals": totals,
            "nrfi_prob": nrfi_probability,
            "rifi_prob": 1.0 - nrfi_probability,
        }
    return GAME_PROP_CACHE[key]


def team_stat_stack(r, team_side="home"):
    home_team = r["home_team"]
    away_team = r["away_team"]
    team = home_team if team_side == "home" else away_team
    opp = away_team if team_side == "home" else home_team
    prob = r["home_win_prob"] if team_side == "home" else r["away_win_prob"]
    model = r["model_prob_home"] if team_side == "home" else (1.0 - r["model_prob_home"])
    pyth = r["pyth_home"] if team_side == "home" else r["pyth_away"]
    form = r["form_home"] if team_side == "home" else r["form_away"]
    sp = r["sp_home"] if team_side == "home" else r["sp_away"]
    bp = r["bp_home"] if team_side == "home" else r["bp_away"]
    lineup = r["home_lineup_strength"] if team_side == "home" else r["away_lineup_strength"]
    market = r["home_market_prob"] if team_side == "home" else r["away_market_prob"]
    edge = r.get("home_edge") if team_side == "home" else r.get("away_edge")
    return {
        "team": team,
        "opp": opp,
        "prob": prob,
        "model": model,
        "pyth": pyth,
        "form": form,
        "sp": sp,
        "bp": bp,
        "lineup": lineup,
        "market": market,
        "edge": edge,
        "signal_quality": r.get("signal_quality", 0.0),
        "model_available": bool(r.get("model_available", False)),
        "sp_available": bool(r.get(f"sp_{team_side}_available", False)),
        "bp_available": int(r.get(f"n_bp_{team_side}", 0) or 0) > 0,
        "form_available": int(r.get(f"n_{team_side}_games", 0) or 0) > 0,
        "pyth_available": bool(r.get(f"pyth_{team_side}_available", False)),
    }


# ================================================================
# PRINT TEAM MATCHUPS
# ================================================================

print()
line()
print("TEAM WIN PREDICTIONS")
line()

for r in results:
    fav = r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"]
    fav_prob = max(r["home_win_prob"], r["away_win_prob"])
    fav_edge = r.get("home_edge") if fav == r["home_team"] else r.get("away_edge")
    ann = pick_annotation(
        fav_prob,
        fav_edge,
        r.get("signal_quality", 0.0),
        market_available=valid(fav_edge),
    )
    opp_team = r["away_team"] if fav == r["home_team"] else r["home_team"]
    team_side = "home" if fav == r["home_team"] else "away"
    signal_summary = team_stat_stack(r, team_side)
    fair_price = fair_american_odds(fav_prob)
    fair_line = f"{fair_price:+d}" if fair_price is not None else "N/A"
    public_price = best_team_price(fav, opp_team)
    public_line = f"{int(public_price):+d}" if valid(public_price) else "N/A"
    edge_line = f"{pct(fav_edge)}" if valid(fav_edge) else "N/A"
    game_label = f"{r['away_team']} @ {r['home_team']}"
    print()
    print(term_dim(f"  +-- {game_label}  ({r['away_pitcher_name']} vs {r['home_pitcher_name']})"))
    print(f"  | {term_bold('PICK')} {term_bold(fav)} {pct(fav_prob)}")
    print(
        f"  | fair {term_purple(fair_line)} | market {term_purple(public_line)} "
        f"| edge {term_purple(edge_line)} | {ann}"
    )

    def component(label, value, available=True):
        return f"{label} {stat_pct(value) if available else 'N/A'}"

    component_text = " | ".join([
        component("model", signal_summary["model"], signal_summary["model_available"]),
        component("pythag", signal_summary["pyth"], signal_summary["pyth_available"]),
        component("form", signal_summary["form"], signal_summary["form_available"]),
        component("starter", signal_summary["sp"], signal_summary["sp_available"]),
        component("bullpen", signal_summary["bp"], signal_summary["bp_available"]),
        component("lineup", signal_summary["lineup"], True),
        component("market", signal_summary["market"], valid(signal_summary["market"])),
    ])
    print(term_dim("  | components: " + component_text))

    metric_card = r.get("metric_cards", {}).get(team_side, {})
    projected_score = metric_card.get("proj_score", {})
    score_label = (
        f"{projected_score.get('away', float('nan')):.1f}-"
        f"{projected_score.get('home', float('nan')):.1f}"
        if valid(projected_score.get("away")) and valid(projected_score.get("home"))
        else "N/A"
    )
    h2h_delta = metric_card.get("h2h_delta")
    h2h_label = f"{h2h_delta * 100:+.1f}pp" if valid(h2h_delta) else "N/A"
    print(
        term_dim(
            "  | card: adjusted " + stat_pct(metric_card.get("ai_probability"))
            + " | base " + stat_pct(metric_card.get("base_projection"))
            + " | EV " + stat_pct(metric_card.get("expected_value"))
            + " | H2H " + h2h_label
            + " | score " + score_label
            + " | margin " + number(metric_card.get("avg_margin"), 2)
        )
    )
    
    # SAFETY NET: Show +1.5/-1.5 lines based on odds data
    safety_net_info = ""
    if odds_data:
        try:
            # Get the total line and see if we can calculate a safety net
            total_line = totals_pick(
                r["home_team"], r["away_team"], pa_df, DATE,
                odds_data=odds_data,
                modern_stats=modern_team_stats
            ).get("market_line")
            
            if valid(total_line):
                # Calculate implied probability for total_line + 1.5 and total_line - 1.5
                total_plus = total_line + 1.5
                total_minus = total_line - 1.5
                
                # Get the model's projected total
                projected_total = r.get("score_profile", {}).get("projected_total")
                
                if valid(projected_total):
                    # Simple safety net: if our projection is significantly different from the line
                    diff_from_line = abs(projected_total - total_line)
                    if diff_from_line >= 1.0:  # If we differ by 1+ runs from the line
                        if projected_total > total_line:
                            safety_net_info = f" | SAFETY: OVER {total_plus} (proj: {projected_total:.1f})"
                        else:
                            safety_net_info = f" | SAFETY: UNDER {total_minus} (proj: {projected_total:.1f})"
        except Exception:
            pass  # Silently fail if safety net calculation doesn't work
    
    print(term_dim("  |" + safety_net_info + "--"))


# ================================================================
# BATTER HIT PREDICTIONS
# ================================================================

metric_reference_context = league_reference_context(pa_df, DATE)

def bvp_shrunk_rate(batter_id, pitcher_id):
    """Batter-vs-pitcher hit rate, shrunk toward league mean by sample size."""
    league_mean = 0.24
    if batter_cache is None:
        return league_mean, 0

    try:
        lookup = BVP_RATE_LOOKUP.get((int(safe_num(batter_id)), int(safe_num(pitcher_id))))
        if lookup:
            hit_rate = safe_num(lookup.get("hit_rate"), league_mean)
            pa = safe_num(lookup.get("plate_appearances", lookup.get("pa")), 0)
            return shrink(hit_rate, pa, league_mean, BVP_SHRINK_K), pa

        if isinstance(batter_cache, pd.DataFrame):
            match = batter_cache[
                (pd.to_numeric(batter_cache["batter"], errors="coerce") == safe_num(batter_id)) &
                (pd.to_numeric(batter_cache["pitcher"], errors="coerce") == safe_num(pitcher_id))
            ]
            if match.empty:
                return league_mean, 0
            hit_rate = safe_num(match.iloc[0].get("hit_rate"), league_mean)
            pa = safe_num(match.iloc[0].get("plate_appearances", match.iloc[0].get("pa")), 0)
            return shrink(hit_rate, pa, league_mean, BVP_SHRINK_K), pa

        if isinstance(batter_cache, dict):
            key = (batter_id, pitcher_id)
            rec = batter_cache.get(key) or batter_cache.get(f"{batter_id}_{pitcher_id}")
            if not rec:
                return league_mean, 0
            pa = safe_num(rec.get("pa", rec.get("plate_appearances")), 0)
            hit_rate = safe_num(rec.get("hit_rate"), league_mean)
            return shrink(hit_rate, pa, league_mean, BVP_SHRINK_K), pa
    except Exception:
        pass

    return league_mean, 0


def _recent_form_cache(as_of_date, lookback_games=10):
    """Build game-level recent-form estimates once for all batters."""
    cache_key = (str(pd.Timestamp(as_of_date).date()), int(lookback_games))
    cached = BATTER_RECENT_FORM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    output = {}
    if pa_df is None or pa_df.empty:
        BATTER_RECENT_FORM_CACHE[cache_key] = output
        return output

    required = {"batter", "game_pk", "game_date", "is_hit"}
    if not required.issubset(pa_df.columns):
        BATTER_RECENT_FORM_CACHE[cache_key] = output
        return output

    ts = pd.Timestamp(as_of_date)
    frame = pa_df.loc[:, ["batter", "game_pk", "game_date", "is_hit"]].copy()
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["is_hit"] = pd.to_numeric(frame["is_hit"], errors="coerce").fillna(0.0)
    frame = frame[
        frame["batter"].notna()
        & frame["game_date"].notna()
        & (frame["game_date"] < ts)
    ]
    if frame.empty:
        BATTER_RECENT_FORM_CACHE[cache_key] = output
        return output

    game_level = (
        frame.groupby(["batter", "game_pk"], as_index=False)
        .agg(game_date=("game_date", "max"), hit=("is_hit", "max"))
        .sort_values(["batter", "game_date"], ascending=[True, False])
    )
    for batter_id, rows in game_level.groupby("batter", sort=False):
        recent = rows.head(int(lookback_games))
        n = len(recent)
        if n == 0:
            continue
        weights = recency_weights(n)
        raw_rate = float(np.dot(weights, recent["hit"].to_numpy(dtype=float)))
        output[int(batter_id)] = (shrink(raw_rate, n, 0.57, 5), n)

    BATTER_RECENT_FORM_CACHE[cache_key] = output
    return output


def batter_recent_form(batter_id, as_of_date, lookback_games=10):
    try:
        batter_key = int(safe_num(batter_id))
    except (TypeError, ValueError):
        batter_key = None
    form = _recent_form_cache(as_of_date, lookback_games).get(batter_key)
    if form is not None:
        return form
    return 0.24, 0


def batter_enhanced_recent_form(batter_id, as_of_date):
    """
    Enhanced recent form that uses ALL available historical data (no date restrictions)
    to maximize predictive power for betting purposes, as requested.
    """
    if pa_df is None:
        return 0.24, 0
    
    try:
        batter_key = int(safe_num(batter_id))
    except (TypeError, ValueError):
        return 0.24, 0
        
    # USE ALL AVAILABLE DATA - NO DATE RESTRICTIONS AS REQUESTED
    batter_pa = pa_df[
        (pd.to_numeric(pa_df["batter"], errors="coerce") == batter_key)
    ].copy()
    
    if batter_pa.empty:
        return 0.24, 0
    
    # Sort by game_date descending to prioritize recent performance
    batter_pa["game_date"] = pd.to_datetime(batter_pa["game_date"], errors="coerce")
    batter_pa = batter_pa.sort_values("game_date", ascending=False)
    
    # Calculate weighted averages for different time windows
    # More recent games get higher weight
    windows = [5, 10, 15, 20, 50, 100]  # Different lookback windows including career totals
    weights = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]  # Weights for each window (must sum to 1.0)
    
    weighted_rates = []
    total_weight = 0
    
    for window, weight in zip(windows, weights):
        window_data = batter_pa.head(window)
        if len(window_data) > 0:
            # Calculate hit rate for this window
            window_hits = window_data["is_hit"].sum()
            window_pa = len(window_data)
            window_rate = window_hits / window_pa if window_pa > 0 else 0.24
            
            # Apply additional recency weighting within the window
            if len(window_data) > 1:
                # Give more weight to recent games within the window
                recency_weights = np.exp(-np.arange(len(window_data)) * 0.05)  # Slower decay for longer windows
                recency_weights = recency_weights / recency_weights.sum()
                
                # Calculate weighted hit rate within window
                weighted_hits = np.dot(window_data["is_hit"].values, recency_weights)
                weighted_pa = np.sum(recency_weights)
                if weighted_pa > 0:
                    window_rate = weighted_hits / weighted_pa
            
            weighted_rates.append(window_rate * weight)
            total_weight += weight
    
    if total_weight > 0:
        final_rate = sum(weighted_rates) / total_weight
    else:
        final_rate = 0.24
        
    # Calculate total PA for confidence
    total_pa = len(batter_pa)
    
    return clamp(final_rate, 0.15, 0.45), total_pa

def pitcher_allowed_profile(opp_pitcher_id, team_name):
    if pitcher_snap is None:
        return {"xba_allowed": 0.24, "k_pct": 0.22, "bb_pct": 0.08, "hard_hit": 0.3}

    try:
        if isinstance(pitcher_snap, pd.DataFrame):
            row = PITCHER_SNAPSHOT_LOOKUP.get(int(safe_num(opp_pitcher_id)))
            if not row:
                row_frame = pitcher_snap[pd.to_numeric(pitcher_snap["pitcher"], errors="coerce") == safe_num(opp_pitcher_id)]
                if not row_frame.empty:
                    row = row_frame.iloc[0].to_dict()
            if not row:
                return {"xba_allowed": 0.24, "k_pct": 0.22, "bb_pct": 0.08, "hard_hit": 0.3}
        else:
            row = pitcher_snap.get(opp_pitcher_id) if isinstance(pitcher_snap, dict) else {}
            if isinstance(row, pd.Series):
                row = row.to_dict()
            if not row:
                return {"xba_allowed": 0.24, "k_pct": 0.22, "bb_pct": 0.08, "hard_hit": 0.3}

        cache_row = PITCHER_MATCHUP_LOOKUP.get(
            (int(safe_num(opp_pitcher_id)), str(team_name).upper())
        ) or {}
        if not cache_row and isinstance(pitcher_cache, pd.DataFrame) and team_name:
            pitch_match = pitcher_cache[
                (pd.to_numeric(pitcher_cache["pitcher"], errors="coerce") == safe_num(opp_pitcher_id)) &
                (pitcher_cache["opposing_team"].astype(str).str.upper() == str(team_name).upper())
            ]
            if not pitch_match.empty:
                cache_row = pitch_match.iloc[0].to_dict()

        # GET PITCHER'S STATCAST ROLLING STATISTICS IF AVAILABLE
        pitch_roll_k_rate = None
        pitch_roll_bb_rate = None
        pitch_roll_xba_against = None
        pitch_roll_hardhit_against = None
        
        if pa_df is not None and opp_pitcher_id is not None:
            try:
                # Get the most recent appearance for this pitcher
                recent_pitcher_pa = pa_df[
                    (pd.to_numeric(pa_df["pitcher"], errors="coerce") == safe_num(opp_pitcher_id)) &
                    (pd.to_datetime(pa_df["game_date"], errors="coerce") < pd.Timestamp(DATE))
                ].sort_values("game_date", ascending=False).head(1)
                
                if not recent_pitcher_pa.empty:
                    # Use pre-computed pitcher rolling statistics from Statcast
                    pitch_roll_k_rate = recent_pitcher_pa.iloc[0].get("pitcher_roll_k_rate")
                    pitch_roll_bb_rate = recent_pitcher_pa.iloc[0].get("pitcher_roll_bb_rate")
                    pitch_roll_xba_against = recent_pitcher_pa.iloc[0].get("pitcher_roll_xba_against")
                    pitch_roll_hardhit_against = recent_pitcher_pa.iloc[0].get("pitcher_roll_hardhit_against")
            except Exception:
                pass  # Will fall back to standard methods below

        # USE STATCAST ROLLING STATISTICS IF AVAILABLE, OTHERWISE FALL BACK TO STANDARD METHODS
        if valid(pitch_roll_k_rate) and valid(pitch_roll_bb_rate) and valid(pitch_roll_xba_against):
            # Use Statcast rolling stats
            k_pct = clamp(pitch_roll_k_rate, 0.05, 0.45)
            bb_pct = clamp(pitch_roll_bb_rate, 0.01, 0.20)
            xba_allowed = clamp(pitch_roll_xba_against, 0.15, 0.45)
            hard_hit = clamp(pitch_roll_hardhit_against, 0.05, 0.60) if valid(pitch_roll_hardhit_against) else 0.30
        else:
            # Fall back to standard methods
            xba_allowed = feature_value(row, "xba_against", "xBA_against", "xBA_allowed")
            if not valid(xba_allowed) and cache_row:
                xba_allowed = safe_num(cache_row.get("xBA_allowed", cache_row.get("xba_allowed")))

            k_pct = feature_value(row, "k_pct", "strikeout_rate")
            if not valid(k_pct) and cache_row:
                k_pct = safe_num(cache_row.get("strikeout_rate", 0.22))

            bb_pct = feature_value(row, "bb_pct", "walk_rate")
            if not valid(bb_pct) and cache_row:
                bb_pct = safe_num(cache_row.get("walk_rate", 0.08))

            hard_hit = feature_value(row, "hard_hit_rate_allowed", "hard_hit_rate_against")
            if not valid(hard_hit) and cache_row:
                hard_hit = safe_num(cache_row.get("hard_hit_rate_allowed", 0.30))

        return {
            "xba_allowed": clamp(xba_allowed, 0.15, 0.45) if valid(xba_allowed) else 0.24,
            "k_pct": clamp(k_pct, 0.05, 0.45) if valid(k_pct) else 0.22,
            "bb_pct": clamp(bb_pct, 0.01, 0.20) if valid(bb_pct) else 0.08,
            "hard_hit": clamp(hard_hit, 0.05, 0.60) if valid(hard_hit) else 0.30,
        }
    except Exception:
        return {"xba_allowed": 0.24, "k_pct": 0.22, "bb_pct": 0.08, "hard_hit": 0.30}


def platoon_lookup(batter_id, team_pitcher_throws):
    if platoon_snap is None:
        return {"hit_rate_vs_hand": 0.24, "xba_vs_hand": 0.24}

    try:
        lookup = PLATOON_LOOKUP.get(
            (int(safe_num(batter_id)), str(team_pitcher_throws).upper())
        )
        if lookup:
            return {
                "hit_rate_vs_hand": safe_num(lookup.get("batter_roll_hit_rate_vs_hand"), 0.24),
                "xba_vs_hand": safe_num(lookup.get("batter_roll_xba_vs_hand"), 0.24),
            }
        if isinstance(platoon_snap, pd.DataFrame):
            row = platoon_snap[
                (pd.to_numeric(platoon_snap["batter"], errors="coerce") == safe_num(batter_id)) &
                (platoon_snap["p_throws"].astype(str).str.upper() == str(team_pitcher_throws).upper())
            ]
            if not row.empty:
                row = row.iloc[0].to_dict()
                return {
                    "hit_rate_vs_hand": safe_num(row.get("batter_roll_hit_rate_vs_hand"), 0.24),
                    "xba_vs_hand": safe_num(row.get("batter_roll_xba_vs_hand"), 0.24),
                }
        elif isinstance(platoon_snap, dict):
            key = (batter_id, team_pitcher_throws)
            val = platoon_snap.get(key)
            if isinstance(val, dict):
                return {
                    "hit_rate_vs_hand": safe_num(val.get("batter_roll_hit_rate_vs_hand"), 0.24),
                    "xba_vs_hand": safe_num(val.get("batter_roll_xba_vs_hand"), 0.24),
                }
    except Exception:
        pass

    return {"hit_rate_vs_hand": 0.24, "xba_vs_hand": 0.24}


def score_batter(batter_id, batter_name, team, opp_pitcher_id, is_home):
    cache_key = (
        int(safe_num(batter_id)) if valid(safe_num(batter_id)) else str(batter_id),
        str(team).upper(),
        int(safe_num(opp_pitcher_id)) if valid(safe_num(opp_pitcher_id)) else str(opp_pitcher_id),
    )
    cached = SCORE_BATTER_CACHE.get(cache_key)
    if cached is not None:
        result = dict(cached)
        result["batter"] = batter_name
        return result

    base_prob = 0.24
    if batter_snap is not None:
        try:
            lookup = BATTER_SNAPSHOT_LOOKUP.get(int(safe_num(batter_id)))
            if lookup:
                row_df = pd.DataFrame([{f: lookup.get(f, np.nan) for f in HIT_FEATURES}])[HIT_FEATURES]
                base_prob = clamp(model_predict_proba(hit_model, row_df)[0][1])
            elif isinstance(batter_snap, pd.DataFrame):
                row = batter_snap[pd.to_numeric(batter_snap["batter"], errors="coerce") == safe_num(batter_id)]
                if not row.empty:
                    row = row.iloc[0].to_dict()
                    row_df = pd.DataFrame([{f: row.get(f, np.nan) for f in HIT_FEATURES}])[HIT_FEATURES]
                    base_prob = clamp(model_predict_proba(hit_model, row_df)[0][1])
            elif isinstance(batter_snap, dict):
                row = batter_snap.get(batter_id)
                if isinstance(row, pd.Series):
                    row = row.to_dict()
                if row:
                    row_df = pd.DataFrame([{f: row.get(f, np.nan) for f in HIT_FEATURES}])[HIT_FEATURES]
                    base_prob = clamp(model_predict_proba(hit_model, row_df)[0][1])
        except Exception:
            pass

    # ENHANCED RECENT FORM: Use sophisticated recent form calculation
    recent_prob, n_recent = batter_enhanced_recent_form(batter_id, DATE)

    pitcher_profile = pitcher_allowed_profile(opp_pitcher_id, team)
    pitcher_allowed = pitcher_profile["xba_allowed"]
    k_pct = pitcher_profile["k_pct"]
    bb_pct = pitcher_profile["bb_pct"]
    hard_hit = pitcher_profile["hard_hit"]

    p_throws = None
    if pa_df is not None and opp_pitcher_id is not None:
        try:
            pd_pitchers = pa_df.loc[pd.to_numeric(pa_df["pitcher"], errors="coerce") == safe_num(opp_pitcher_id), "p_throws"]
            if isinstance(pd_pitchers, pd.Series) and not pd_pitchers.empty:
                non_null = pd_pitchers.dropna()
                if not non_null.empty:
                    p_throws = str(non_null.iloc[0]).upper()
        except Exception:
            p_throws = None

    # ENHANCED: Use pre-computed rolling statistics from Statcast
    platoon_prob = 0.24  # Default
    park_factor = 1.0    # Default
    
    if pa_df is not None and batter_id is not None:
        try:
            # Get the most recent plate appearance for this batter
            recent_batter_pa = pa_df[
                (pd.to_numeric(pa_df["batter"], errors="coerce") == safe_num(batter_id)) &
                (pd.to_datetime(pa_df["game_date"], errors="coerce") < pd.Timestamp(DATE))
            ].sort_values("game_date", ascending=False).head(1)
            
            if not recent_batter_pa.empty:
                # USE PRE-COMPUTED STATCAST ROLLING STATISTICS FOR ENHANCED PREDICTION
                # Batter's rolling hit rate (overall) - direct measure of recent hitting ability
                roll_hit_rate = recent_batter_pa.iloc[0].get("batter_roll_hit_rate")
                # Batter's rolling expected batting average - better indicator of true talent
                roll_xba = recent_batter_pa.iloc[0].get("batter_roll_xba")
                # Batter's rolling hard hit rate - correlates strongly with future success
                roll_hard_hit = recent_batter_pa.iloc[0].get("batter_roll_hardhit_rate")
                
                # Use pre-computed rolling hit rate vs hand (already implemented for platoon)
                roll_vs_hand = recent_batter_pa.iloc[0].get("batter_roll_hit_rate_vs_hand")
                if valid(roll_vs_hand):
                    platoon_prob = clamp(roll_vs_hand, 0.15, 0.45)
                
                # ENHANCE RECENT FORM WITH STATCAST ROLLING METRICS
                # If we have strong Statcast rolling data, we can adjust our recent form calculation
                statcast_weight = 0.0
                statcast_adjustment = 0.0
                
                if valid(roll_hit_rate):
                    # Blend our calculated recent form with Statcast rolling hit rate
                    statcast_weight += 0.4
                    statcast_adjustment += roll_hit_rate * 0.4
                    
                if valid(roll_xba):
                    # xBA is often more predictive than actual BA
                    statcast_weight += 0.3
                    statcast_adjustment += roll_xba * 0.3
                    
                if valid(roll_hard_hit):
                    # Hard hit rate is a leading indicator of future batting success
                    # Convert hard hit rate to expected hit rate (rough approximation)
                    # Typically, hard hit rate around 0.300 corresponds to hit rate around 0.280
                    estimated_hit_rate_from_hard_hit = clamp(roll_hard_hit * 0.93, 0.15, 0.45)
                    statcast_weight += 0.3
                    statcast_adjustment += estimated_hit_rate_from_hard_hit * 0.3
                
                # Apply Statcast enhancement if we have sufficient data
                if statcast_weight > 0:
                    statcast_enhanced_rate = statcast_adjustment / statcast_weight
                    # Blend Statcast enhancement with our calculated recent form
                    recent_prob = clamp((recent_prob * 0.6) + (statcast_enhanced_rate * 0.4), 0.15, 0.45)
                
                # Use pre-computed park factor from the batter's recent game
                # We need to determine if the batter was home or away in that game
                batter_home_team = recent_batter_pa.iloc[0].get("home_team")
                batter_away_team = recent_batter_pa.iloc[0].get("away_team")
                team_is_home = (batter_home_team == team) if batter_home_team else False
                
                if team_is_home and batter_home_team:
                    park_factor_val = recent_batter_pa.iloc[0].get("park_hit_factor")
                elif not team_is_home and batter_away_team:
                    # For away team, we need the home team's park factor
                    park_factor_val = recent_batter_pa.iloc[0].get("park_hit_factor")
                else:
                    park_factor_val = None
                    
                if valid(park_factor_val):
                    park_factor = clamp(park_factor_val, 0.7, 1.3)
                # USE ADDITIONAL PARK FACTORS
                # park_hr_factor could be useful for predicting home run props
                # but we're focused on hit probability for now
            else:
                # Fallback to platoon lookup
                platoon = platoon_lookup(batter_id, p_throws) if p_throws else {"hit_rate_vs_hand": 0.24, "xba_vs_hand": 0.24}
                platoon_raw = safe_num(platoon.get("hit_rate_vs_hand"), 0.24)
                platoon_prob = shrink(platoon_raw, 40, 0.24, 20) if valid(platoon_raw) else 0.24
                
                # Fallback to park_snap
                if park_snap is not None:
                    try:
                        if isinstance(park_snap, pd.DataFrame):
                            row = park_snap[park_snap["home_team"].astype(str).str.upper() == str(team).upper()]
                            if not row.empty:
                                park_factor = safe_num(row.iloc[0].get("park_hit_factor"), 1.0)
                        elif isinstance(park_snap, dict):
                            park_factor = safe_num(park_snap.get(team, {}).get("park_hit_factor"), 1.0)
                    except Exception:
                        pass
        except Exception:
            # Fallback to standard methods
            platoon = platoon_lookup(batter_id, p_throws) if p_throws else {"hit_rate_vs_hand": 0.24, "xba_vs_hand": 0.24}
            platoon_raw = safe_num(platoon.get("hit_rate_vs_hand"), 0.24)
            platoon_prob = shrink(platoon_raw, 40, 0.24, 20) if valid(platoon_raw) else 0.24
            
            if park_snap is not None:
                try:
                    if isinstance(park_snap, pd.DataFrame):
                        row = park_snap[park_snap["home_team"].astype(str).str.upper() == str(team).upper()]
                        if not row.empty:
                            park_factor = safe_num(row.iloc[0].get("park_hit_factor"), 1.0)
                    elif isinstance(park_snap, dict):
                        park_factor = safe_num(park_snap.get(team, {}).get("park_hit_factor"), 1.0)
                except Exception:
                    pass
    else:
        # Fallback to standard methods when pa_df is not available
        platoon = platoon_lookup(batter_id, p_throws) if p_throws else {"hit_rate_vs_hand": 0.24, "xba_vs_hand": 0.24}
        platoon_raw = safe_num(platoon.get("hit_rate_vs_hand"), 0.24)
        platoon_prob = shrink(platoon_raw, 40, 0.24, 20) if valid(platoon_raw) else 0.24
        
        if park_snap is not None:
            try:
                if isinstance(park_snap, pd.DataFrame):
                    row = park_snap[park_snap["home_team"].astype(str).str.upper() == str(team).upper()]
                    if not row.empty:
                        park_factor = safe_num(row.iloc[0].get("park_hit_factor"), 1.0)
                elif isinstance(park_snap, dict):
                    park_factor = safe_num(park_snap.get(team, {}).get("park_hit_factor"), 1.0)
            except Exception:
                pass

    # ENSURE PARK_FACTOR IS DEFINED AND CONVERT TO PROBABILITY
    park_factor = clamp(park_factor, 0.7, 1.3) if valid(park_factor) else 1.0
    # Park factors are multipliers around 1.0, not probabilities. Convert the
    # multiplier to a modest hit-rate signal before using it in logit space.
    park_prob = clamp(0.24 * park_factor, 0.15, 0.45)

    # ENHANCED: Use platoon advantage from Statcast when available
    platoon_advantage_prob = None
    if pa_df is not None and batter_id is not None and opp_pitcher_id is not None:
        try:
            # Look for recent matchups between this batter and pitcher
            recent_matchup = pa_df[
                (pd.to_numeric(pa_df["batter"], errors="coerce") == safe_num(batter_id)) &
                (pd.to_numeric(pa_df["pitcher"], errors="coerce") == safe_num(opp_pitcher_id)) &
                (pd.to_datetime(pa_df["game_date"], errors="coerce") < pd.Timestamp(DATE))
            ].sort_values("game_date", ascending=False).head(10)  # Last 10 matchups
            
            if not recent_matchup.empty:
                # Use the average platoon advantage from recent matchups
                avg_platoon_advantage = recent_matchup["platoon_advantage"].mean()
                if valid(avg_platoon_advantage):
                    # Convert platoon advantage to a probability adjustment
                    # platoon_advantage is typically positive for advantage, negative for disadvantage
                    platoon_advantage_prob = clamp(0.24 + (avg_platoon_advantage * 0.15), 0.15, 0.45)
        except Exception:
            pass  # Will fall back to standard platoon lookup

    # Use platoon advantage if available, otherwise use our calculated platoon_prob
    if platoon_advantage_prob is not None:
        platoon_prob = platoon_advantage_prob

    bvp_prob, bvp_n = bvp_shrunk_rate(batter_id, opp_pitcher_id)

    # ENHANCED WEIGHTING: Adjust weights to prioritize factors that most directly impact betting outcomes
    # Based on user request to "use arb or historical whatever would be best to win the bets"
    blended = (
        WB_MODEL * logit(base_prob)                           # 0.40 weight - trained model (reduced slightly)
        + WB_RECENT_FORM * logit(clamp(recent_prob))          # 0.25 weight - enhanced recent form 
        + WB_PITCHER_ALLOWED * logit(clamp(pitcher_allowed))  # 0.15 weight - pitcher allowed stats
        + WB_PLATOON * logit(clamp(platoon_prob))             # 0.10 weight - platoon factors
        + WB_PARK * logit(clamp(park_prob))                   # 0.08 weight - park factors
        + WB_BVP * logit(clamp(bvp_prob))                     # 0.02 weight - BvP historical data
    )
    final_prob = inv_logit(blended)

    result = {
        "batter": batter_name,
        "team": team,
        "final_prob": final_prob,
        "base_model": base_prob,
        "recent_form": recent_prob,
        "n_recent": n_recent,
        "bvp": bvp_prob,
        "bvp_n": bvp_n,
        "pitcher_xba_allowed": pitcher_allowed,
        "pitcher_k_pct": k_pct,
        "pitcher_bb_pct": bb_pct,
        "pitcher_hard_hit": hard_hit,
        "platoon_hit_rate_vs_hand": platoon_prob,
        "park_hit_factor": park_factor,
    }
    SCORE_BATTER_CACHE[cache_key] = dict(result)
    return result


def resolve_batter_name(batter_id, *, allow_remote=True):
    """Best-effort human name resolver from batter snapshot/cache data."""
    if batter_id is None:
        return "Unknown hitter"
    batter_id_num = safe_num(batter_id)
    if isinstance(batter_snap, pd.DataFrame):
        try:
            df = batter_snap.copy()
            id_col = None
            for candidate in ["batter", "batter_id", "player_id"]:
                if candidate in df.columns:
                    id_col = candidate
                    break
            if id_col is not None:
                mask = pd.to_numeric(df[id_col], errors="coerce") == batter_id_num
                if mask.any():
                    row = df.loc[mask].iloc[0].to_dict()
                    for key in ["batter_name", "player_name", "name", "full_name"]:
                        if key in row and str(row[key]).strip():
                            return str(row[key]).strip()
        except Exception:
            pass
    elif isinstance(batter_snap, dict):
        try:
            row = batter_snap.get(batter_id) or batter_snap.get(int(batter_id_num)) if valid(batter_id_num) else None
            if isinstance(row, pd.Series):
                row = row.to_dict()
            if isinstance(row, dict):
                for key in ["batter_name", "player_name", "name", "full_name"]:
                    if key in row and str(row[key]).strip():
                        return str(row[key]).strip()
        except Exception:
            pass
    if valid(batter_id_num) and int(batter_id_num) in PLAYER_NAME_CACHE:
        return PLAYER_NAME_CACHE[int(batter_id_num)]
    if allow_remote:
        try:
            resp = requests.get(f"https://statsapi.mlb.com/api/v1/people/{int(batter_id_num)}", timeout=15)
            if resp.status_code == 200:
                person = (resp.json().get("people") or [{}])[0]
                full_name = person.get("fullName") or person.get("firstName")
                if full_name:
                    PLAYER_NAME_CACHE[int(batter_id_num)] = str(full_name)
                    return str(full_name)
        except Exception:
            pass
    return f"Batter {int(batter_id_num) if valid(batter_id_num) else batter_id}"


def team_recent_batter_candidates(team, opponent_pitcher_id, as_of_date, limit=8):
    """Return most relevant hitters for a team and matchup, using ALL available historical data 
    (no date restrictions) to maximize predictive power for betting as requested."""
    if pa_df is None:
        return []
    
    # USE ALL AVAILABLE DATA - NO DATE RESTRICTIONS AS REQUESTED
    recent = pa_df[
        (pa_df["batting_team"].astype(str).str.upper() == str(team).upper())
    ].copy()

    if recent.empty:
        return []

    if "batter" not in recent.columns:
        return []

    recent = recent.dropna(subset=["batter"]).copy()
    recent["batter"] = pd.to_numeric(recent["batter"], errors="coerce")
    recent = recent[recent["batter"].notna()].copy()

    # Sort by game_date descending to prioritize more recent performance when samples are equal
    recent["game_date"] = pd.to_datetime(recent["game_date"], errors="coerce")
    recent = recent.sort_values("game_date", ascending=False)

    agg = (
        recent.groupby("batter")
        .agg(pa=("batter", "size"), hit_rate=("is_hit", "mean"), recent_hits=("is_hit", "sum"))
        .reset_index()
    )
    if agg.empty:
        return []
    
    # Sort by PA volume (primary) and hit rate (secondary) to get most reliable hitters
    agg = agg.sort_values(["pa", "hit_rate"], ascending=[False, False]).head(limit * 3)
    out = []
    # Get roster IDs to prioritize current team members
    roster_ids = recent_batter_ids_for_team(team, limit=limit*3)
    roster_set = set(roster_ids)
    for _, row in agg.iterrows():
        batter_id = int(row["batter"])
        # Prioritize current roster members but don't exclude others completely
        priority_boost = 1.2 if (roster_set and batter_id in roster_set) else 1.0
        p = score_batter(batter_id, resolve_batter_name(batter_id), team, opponent_pitcher_id, is_home=False)
        # Apply priority boost to final probability for sorting
        boosted_prob = min(0.99, p["final_prob"] * priority_boost)
        out.append({
            "batter_id": batter_id,
            "batter_name": resolve_batter_name(batter_id),
            "final_prob": p["final_prob"],  # Keep original for display
            "boosted_prob": boosted_prob,   # Use for sorting
            "recent_form": p["recent_form"],
            "bvp": p["bvp"],
            "pitcher_xba_allowed": p["pitcher_xba_allowed"],
            "pitcher_k_pct": p["pitcher_k_pct"],
            "pitcher_bb_pct": p["pitcher_bb_pct"],
            "platoon_hit_rate_vs_hand": p["platoon_hit_rate_vs_hand"],
            "pa": int(row["pa"]),
            "hit_rate": float(row["hit_rate"]),
            "recent_hits": int(row["recent_hits"]),
        })
    # Sort by boosted probability (which favors roster members) then actual probability
    out = sorted(out, key=lambda x: (x["boosted_prob"], x["final_prob"]), reverse=True)[:limit]
    # Remove the boosted_prob field before returning
    for item in out:
        if "boosted_prob" in item:
            del item["boosted_prob"]
    return out



def team_pick_reason(r, team_side="home"):
    home_team = r["home_team"]
    away_team = r["away_team"]
    team = home_team if team_side == "home" else away_team
    opp = away_team if team_side == "home" else home_team
    prob = r["home_win_prob"] if team_side == "home" else r["away_win_prob"]
    model = r["model_prob_home"] if team_side == "home" else (1.0 - r["model_prob_home"])
    form = r["form_home"] if team_side == "home" else r["form_away"]
    sp = r["sp_home"] if team_side == "home" else r["sp_away"]
    bp = r["bp_home"] if team_side == "home" else r["bp_away"]
    lineup = r["home_lineup_strength"] if team_side == "home" else r["away_lineup_strength"]
    market = r["home_market_prob"] if team_side == "home" else r["away_market_prob"]
    edge = r.get("home_edge") if team_side == "home" else r.get("away_edge")
    pyth = r["pyth_home"] if team_side == "home" else r["pyth_away"]
    n_games = r.get("n_home_games", 0) if team_side == "home" else r.get("n_away_games", 0)
    sp_available = r.get(f"sp_{team_side}_available", False)
    pyth_available = r.get(f"pyth_{team_side}_available", False)
    n_bp = int(r.get(f"n_bp_{team_side}", 0) or 0)

    parts = [f"{team} leads {opp} on the available pregame evidence at {pct(prob)}."]
    if r.get("model_available") and valid(model):
        parts.append(f"The trained win model contributes {pct(model)} before the composite adjustment.")
    else:
        parts.append("The trained win model was unavailable for this row, so its component was set to neutral rather than guessed.")
    if valid(form) and n_games:
        parts.append(f"The recent run-form component is {number(form, 3)} from {int(n_games)} completed games.")
    if sp_available and valid(sp) and abs(sp - 0.5) >= 0.02:
        parts.append(f"The available starter component is {number(sp, 3)} for {team}.")
    if valid(bp) and n_bp:
        parts.append(f"The bullpen component is {number(bp, 3)} using {n_bp} recent bullpen games.")
    if pyth_available and valid(pyth):
        parts.append(f"Current-season or historical run prevention supports a Pythagorean estimate of {pct(pyth)}.")
    if valid(lineup) and abs(lineup - 0.5) >= 0.02:
        parts.append(f"The projected lineup component is {number(lineup, 3)} against the opposing starter.")
    if valid(market) and valid(edge):
        parts.append(f"The no-vig market estimate is {pct(market)}, leaving a model edge of {pct(edge)}.")
    else:
        parts.append("No usable market price was available, so this is a model lean rather than a value claim.")
    return clean_sentence(" ".join(parts[:6]))


def batter_matchup_score(batter_pick):
    """Batter-side matchup score. A weak opposing starter who allows more contact is a better hitter spot."""
    base = safe_num(batter_pick.get("final_prob"), 0.24)
    recent = safe_num(batter_pick.get("recent_form"), 0.24)
    bvp = safe_num(batter_pick.get("bvp"), 0.24)
    platoon = safe_num(batter_pick.get("platoon_hit_rate_vs_hand"), 0.24)
    opp_allowed = safe_num(batter_pick.get("pitcher_xba_allowed"), 0.24)
    park_factor = safe_num(batter_pick.get("park_hit_factor"), 1.0)
    park_signal = clamp(0.24 * park_factor, 0.15, 0.45)
    components = [
        (0.32, clamp(base)),
        (0.18, clamp(recent)),
        (0.15, clamp(bvp)),
        (0.12, clamp(platoon)),
        (0.18, clamp(opp_allowed)),
        (0.05, park_signal),
    ]
    total_weight = sum(weight for weight, value in components if valid(value))
    if total_weight <= 0:
        return 0.24
    return sum(weight * value for weight, value in components if valid(value)) / total_weight


def batter_stat_stack(batter_pick):
    return {
        "base": batter_pick.get("base_model", 0.24),
        "recent_form": batter_pick.get("recent_form", 0.24),
        "bvp": batter_pick.get("bvp", 0.24),
        "platoon": batter_pick.get("platoon_hit_rate_vs_hand", 0.24),
        "pitcher_xba": batter_pick.get("pitcher_xba_allowed", 0.24),
        "pitcher_k": batter_pick.get("pitcher_k_pct", 0.22),
        "pitcher_bb": batter_pick.get("pitcher_bb_pct", 0.08),
        "hard_hit": batter_pick.get("pitcher_hard_hit", 0.30),
        "park": batter_pick.get("park_hit_factor", 0.5),
        "final_prob": batter_pick.get("final_prob", 0.24),
        "matchup_score": batter_pick.get("matchup_score", 0.24),
    }


def batter_pick_reason(batter_pick):
    name = batter_pick.get("batter", "Unknown hitter")
    team = batter_pick.get("team", "unknown team")
    probability = batter_pick.get("final_prob")
    recent_games = int(batter_pick.get("n_recent", 0) or 0)
    bvp_n = int(batter_pick.get("bvp_n", 0) or 0)
    parts = [
        f"{name} ({team}) has an adjusted game-hit estimate of {pct(probability)}.",
        f"The recent game-hit rate is {number(batter_pick.get('recent_form'), 3)} from {recent_games} completed games.",
        f"The opposing starter profile allows approximately {number(batter_pick.get('pitcher_xba_allowed'), 3)} xBA, "
        f"with a {number(batter_pick.get('pitcher_k_pct'), 3)} strikeout rate and {number(batter_pick.get('pitcher_bb_pct'), 3)} walk rate.",
    ]
    if bvp_n:
        parts.append(f"The BvP input is shrunk from {bvp_n} historical PA, so the sample influences the estimate without being treated as proof.")
    else:
        parts.append("There is no usable BvP sample, so that component stays at the league prior.")
    if batter_pick.get("modern_available"):
        parts.append(f"Current-season MLB data is available for {int(batter_pick.get('modern_games', 0) or 0)} games; its PA hit rate is kept separate from the game-hit probability.")
    else:
        parts.append("Current-season player data is unavailable or the local Statcast file is stale; no high-confidence claim is made from old form alone.")
    return clean_sentence(" ".join(parts[:6]))


# ================================================================
# FINAL RANKED PICKS
# ================================================================

def pick_rank_key(r, side):
    edge = r.get(f"{side}_edge")
    prob = r.get(f"{side}_win_prob")
    return edge if valid(edge) else (prob - 0.5)

all_picks = []
for r in results:
    all_picks.append((pick_rank_key(r, "home"), r["home_team"], r["away_team"], r["home_win_prob"], r.get("home_edge"), r))
    all_picks.append((pick_rank_key(r, "away"), r["away_team"], r["home_team"], r["away_win_prob"], r.get("away_edge"), r))

all_picks.sort(key=lambda x: x[0], reverse=True)
top_2_win_picks = []
for rank_val, team, opp, prob, edge, r in all_picks:
    signal_quality = r.get("signal_quality", 1.0)
    base_prob = prob if team == r["home_team"] else (1.0 - prob)
    if signal_quality < 0.3:
        continue
    if not valid(edge) and base_prob < 0.54:
        continue
    if valid(edge) and edge < 0.01 and base_prob < 0.56:
        continue
    top_2_win_picks.append((rank_val, team, opp, prob, edge, r))
    if len(top_2_win_picks) >= 2:
        break

print()
line()
print("TOP 2 WIN PICKS OF THE DAY")
line()
for rank_val, team, opp, prob, edge, r in top_2_win_picks:
    market_probability = (
        r.get("home_market_prob") if team == r["home_team"]
        else r.get("away_market_prob")
    )
    is_upset = (
        valid(edge)
        and valid(market_probability)
        and market_probability < 0.5
        and prob > 0.5
        and edge > 0.02
    )
    label = "UPSET VALUE" if is_upset else pick_annotation(
        prob,
        edge,
        r.get("signal_quality", 0.0),
        market_available=valid(edge),
    )
    fair_price = fair_american_odds(prob)
    market_price = best_team_price(team, opp)
    market_line = f"{int(market_price):+d}" if valid(market_price) else "N/A"
    edge_str = f", edge {pct(edge)}" if valid(edge) else ""
    print(f"  {team} over {opp} - {pct(prob)} [{label}] | fair {fair_price:+d} | market {market_line}{edge_str}")
    print(f"      Why: {team_pick_reason(r, 'home' if team == r['home_team'] else 'away')}")

print()
line()
print("TOP 2 BATTER HIT PICKS OF THE DAY")
line()

# Score hitters across the full slate instead of only the favorite team. This keeps
# the batter section from being artificially limited to a single side when a better
# hitter spot exists elsewhere on the board.
all_batter_candidates = []
seen_candidates = set()
for game in games:
    for team in [game["home_team"], game["away_team"]]:
        opponent_pitcher_id = game.get("away_pitcher_id") if team == game["home_team"] else game.get("home_pitcher_id")
        if not opponent_pitcher_id:
            continue
        for batter_id in recent_batter_ids_for_team(team, limit=12):
            if batter_id is None:
                continue
            key = (team, int(batter_id))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            # Do not make one remote player-name request per candidate.
            # Names are resolved remotely only for the two final displayed picks.
            name = resolve_batter_name(batter_id, allow_remote=False)
            batter_prob = score_batter(batter_id, name, team, opponent_pitcher_id, is_home=(team == game["home_team"]))
            batter_prob["raw_final_prob"] = batter_prob.get("final_prob")
            modern_player = modern_player_stats.get(int(batter_id), {}) or {}
            modern_available = bool(modern_player.get("available"))
            local_fresh = DATA_STALENESS_DAYS is None or DATA_STALENESS_DAYS <= 14
            player_signal_quality = 0.72 if modern_available else (0.48 if local_fresh else 0.22)
            if int(batter_prob.get("n_recent", 0) or 0) < 5:
                player_signal_quality *= 0.75
            adjusted_prob, evidence_factor = confidence_adjusted_probability(
                batter_prob.get("final_prob"),
                signal_quality=player_signal_quality,
                modern_games=modern_player.get("games_played", 0),
                local_staleness_days=DATA_STALENESS_DAYS,
            )
            if valid(adjusted_prob):
                batter_prob["final_prob"] = adjusted_prob
            batter_prob["confidence_evidence"] = evidence_factor
            batter_prob["player_signal_quality"] = player_signal_quality
            batter_prob["modern_available"] = modern_available
            batter_prob["modern_games"] = int(modern_player.get("games_played", 0) or 0)
            batter_prob["modern_pa"] = modern_player.get("plate_appearances")
            batter_prob["eligible_for_display"] = bool(
                (modern_available and int(modern_player.get("games_played", 0) or 0) >= 5)
                or (local_fresh and int(batter_prob.get("n_recent", 0) or 0) >= 8)
            )
            batter_prob["matchup_score"] = batter_matchup_score(batter_prob)
            batter_prob["batter_id"] = int(batter_id)
            batter_prob["team"] = team
            batter_prob["opp_pitcher_id"] = opponent_pitcher_id
            all_batter_candidates.append(batter_prob)

all_batter_candidates = sorted(all_batter_candidates, key=lambda x: (x["matchup_score"], x["final_prob"]), reverse=True)

selected_batters = []
seen_names = set()
for batter_pick in all_batter_candidates:
    key = (batter_pick["batter"], batter_pick["team"])
    if key in seen_names:
        continue
    seen_names.add(key)
    # Always consider batters for top 2 - we'll sort by mathematical score later
    selected_batters.append(batter_pick)
    if len(selected_batters) >= 50:  # Take a reasonable pool to choose from
        break

# Sort by our mathematical score (matchup_score primary, final_prob secondary) and take top 2
selected_batters = sorted(selected_batters, key=lambda x: (x.get("matchup_score", 0), x.get("final_prob", 0)), reverse=True)
top_2_batter_picks = selected_batters[:2]

# Resolve names remotely only after ranking. This keeps a slow or unavailable
# people endpoint from blocking the entire slate while preserving readable names
# for the small number of displayed picks.
for batter_pick in top_2_batter_picks:
    batter_pick["batter"] = resolve_batter_name(
        batter_pick.get("batter_id"), allow_remote=True
    )

# Build expensive history/H2H cards only for the final displayed picks. The
# model still scores the whole slate, but we do not rescan 565k PA rows for
# every roster candidate that will never be shown.
for batter_pick in top_2_batter_picks:
    batter_pick["metric_card"] = build_batter_metric_card(
        batter_id=batter_pick.get("batter_id"),
        batter_name=batter_pick.get("batter", "Unknown hitter"),
        team=batter_pick.get("team", ""),
        opponent_pitcher_id=batter_pick.get("opp_pitcher_id"),
        base_projection=batter_pick.get("base_model"),
        # Pass the raw composite once; the card applies the same confidence
        # adjustment used for ranking, avoiding double shrinkage.
        final_probability=batter_pick.get("raw_final_prob", batter_pick.get("final_prob")),
        pa_df=pa_df,
        as_of_date=DATE,
        local_staleness_days=DATA_STALENESS_DAYS,
        signal_quality=batter_pick.get("player_signal_quality", 0.35),
        modern_player=modern_player_stats.get(int(batter_pick["batter_id"]), {}),
        reference_context=metric_reference_context,
    )

# Always show top 2 batter picks based on mathematical score
if top_2_batter_picks:
    for ix, batter_pick in enumerate(top_2_batter_picks, start=1):
        player = batter_pick["batter"]
        team = batter_pick["team"]
        opp_pitcher = batter_pick["opp_pitcher_id"]
        label = pick_annotation(
            batter_pick["final_prob"],
            signal_quality=batter_pick.get("player_signal_quality", 0.0),
            market_available=False,
        )
        fair_price = fair_american_odds(batter_pick['final_prob'])
        public_price = best_team_price(team, next((g['home_team'] if g['away_team'] == team else g['away_team'] for g in games if team in [g['home_team'], g['away_team']]), team))
        if not valid(public_price):
            public_price_line = "market N/A"
        else:
            public_price_line = f"market {int(public_price):+d}"
        pitcher_label = pitcher_name_for_id(opp_pitcher)
        print(f"  {ix}. {player} ({team}) vs {pitcher_label} - {pct(batter_pick['final_prob'])} [{label}] | fair {fair_price:+d} | {public_price_line}")
        print(f"      Why: {batter_pick_reason(batter_pick)}")
        batter_card = batter_pick.get("metric_card", {})
        h2h_delta = batter_card.get("h2h_delta")
        h2h_label = f"{h2h_delta * 100:+.1f}pp" if valid(h2h_delta) else "N/A"
        modern = batter_card.get("modern_context", {})
        modern_label = (
            f"{modern.get('pa_hit_rate') * 100:.1f}% PA"
            if valid(modern.get("pa_hit_rate")) else "N/A"
        )
        print(
            "      Card: AI " + stat_pct(batter_card.get("ai_probability"))
            + " | base " + stat_pct(batter_card.get("base_projection"))
            + " | L5 " + stat_pct(batter_card.get("last5"))
            + " | L10 " + stat_pct(batter_card.get("last10"))
            + " | H2H " + h2h_label
            + " | usage " + number(batter_card.get("usage_pa_per_game"), 2)
            + " PA/G | impact " + number(batter_card.get("impact_rtg"), 1)
            + " | modern " + modern_label
        )
else:
    # Fallback: show top 2 anyway even if they don't meet thresholds
    if selected_batters:
        fallback_picks = sorted(selected_batters, key=lambda x: (x.get("matchup_score", 0), x.get("final_prob", 0)), reverse=True)[:2]
        for ix, batter_pick in enumerate(fallback_picks, start=1):
            player = batter_pick["batter"]
            team = batter_pick["team"]
            opp_pitcher = batter_pick["opp_pitcher_id"]
            label = "MATH TOP"  # Indicate this is selected purely by mathematical score
            fair_price = fair_american_odds(batter_pick['final_prob'])
            public_price = best_team_price(team, next((g['home_team'] if g['away_team'] == team else g['away_team'] for g in games if team in [g['home_team'], g['away_team']]), team))
            if not valid(public_price):
                public_price_line = "market N/A"
            else:
                public_price_line = f"market {int(public_price):+d}"
            pitcher_label = pitcher_name_for_id(opp_pitcher)
            print(f"  {ix}. {player} ({team}) vs {pitcher_label} - {pct(batter_pick['final_prob'])} [{label}] | fair {fair_price:+d} | {public_price_line}")
            print(f"      Why: {batter_pick_reason(batter_pick)}")
            batter_card = batter_pick.get("metric_card", {})
            h2h_delta = batter_card.get("h2h_delta")
            h2h_label = f"{h2h_delta * 100:+.1f}pp" if valid(h2h_delta) else "N/A"
            modern = batter_card.get("modern_context", {})
            modern_label = (
                f"{modern.get('pa_hit_rate') * 100:.1f}% PA"
                if valid(modern.get("pa_hit_rate")) else "N/A"
            )
            print(
                "      Card: AI " + stat_pct(batter_card.get("ai_probability"))
                + " | base " + stat_pct(batter_card.get("base_projection"))
                + " | L5 " + stat_pct(batter_card.get("last5"))
                + " | L10 " + stat_pct(batter_card.get("last10"))
                + " | H2H " + h2h_label
                + " | usage " + number(batter_card.get("usage_pa_per_game"), 2)
                + " PA/G | impact " + number(batter_card.get("impact_rtg"), 1)
                + " | modern " + modern_label
            )
    else:
        print("  No batter data available for selection.")

print()
line()
print("FIRST-INNING AND TOTALS MARKET")
line()
for game in games:
    home_team = game["home_team"]
    away_team = game["away_team"]
    prop_bundle = game_prop_bundle(home_team, away_team)
    nrfi_prob = prop_bundle["nrfi_prob"]
    rifi_prob = prop_bundle["rifi_prob"]
    totals = prop_bundle["totals"]
    over_price = best_total_market(home_team, away_team, odds_data, side="over")
    under_price = best_total_market(home_team, away_team, odds_data, side="under")
    fair_over = fair_american_odds(totals["model_over_prob"])
    fair_under = fair_american_odds(totals["model_under_prob"])
    over_label = {"over": "OVER", "under": "UNDER"}.get(totals.get("pick"), "PASS")
    market_over = f"{int(over_price['price']):+d}" if over_price else "N/A"
    market_under = f"{int(under_price['price']):+d}" if under_price else "N/A"
    projected_total = totals.get("projected_total")
    projected_total_label = f"{projected_total:.1f}" if valid(projected_total) else "N/A"
    market_line = totals.get("market_line")
    market_line_label = f"{market_line:.1f}" if valid(market_line) else "N/A"
    coverage_label = f"coverage {totals.get("data_coverage", 0.0) * 100:.0f}% / modern {totals.get("modern_weight", 0.0) * 100:.0f}%"
    print(f"  {away_team} @ {home_team} | NRFI {pct(nrfi_prob)} | RIFI {pct(rifi_prob)} | total proj {projected_total_label} | {over_label} {market_line_label} | fair {fair_over:+d}/{fair_under:+d} | {coverage_label} | market over {market_over} / under {market_under}")
    print(f"      Why: {prop_reason_under(totals)} {prop_reason_nrfi({'prob': nrfi_prob})}")
print()
line()
def write_prediction_report():
    """Write the same transparent cards to a local, secret-free JSON file."""
    report_path = os.getenv(
        "PREDICTION_REPORT_PATH",
        os.path.join(ROOT, "prediction_report.json"),
    )
    report = {
        "prediction_date": DATE,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "data_coverage": {
            "local_statcast_earliest": str(RAW_EARLIEST_DATE.date()) if pd.notna(RAW_EARLIEST_DATE) else None,
            "local_statcast_latest": str(RAW_LATEST_DATE.date()) if pd.notna(RAW_LATEST_DATE) else None,
            "local_staleness_days": DATA_STALENESS_DAYS,
            "modern_team_count": len(modern_team_stats),
            "modern_player_count": sum(bool(row.get("available")) for row in modern_player_stats.values()),
        },
        "metric_definitions": metric_definitions(),
        "games": [
            {
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "home_pitcher": r.get("home_pitcher_name"),
                "away_pitcher": r.get("away_pitcher_name"),
                "score_profile": r.get("score_profile"),
                "props": GAME_PROP_CACHE.get(
                    (str(r.get("home_team", "")).upper(), str(r.get("away_team", "")).upper()),
                    {},
                ),
                "home_card": r.get("metric_cards", {}).get("home"),
                "away_card": r.get("metric_cards", {}).get("away"),
                "raw_result": {
                    key: r.get(key) for key in [
                        "home_win_prob", "away_win_prob", "model_prob_home", "model_available",
                        "home_edge", "away_edge", "signal_quality",
                        "h2h_l5", "h2h_l10", "h2h_delta", "h2h_games",
                    ]
                },
            }
            for r in results
        ],
        "top_win_picks": [
            {
                "team": team, "opponent": opp, "probability": prob,
                "edge": edge,
                "metric_card": r.get("metric_cards", {}).get(
                    "home" if team == r.get("home_team") else "away"
                ),
            }
            for _, team, opp, prob, edge, r in top_2_win_picks
        ],
        "top_batter_picks": [
            {
                "batter": pick.get("batter"),
                "team": pick.get("team"),
                "probability": pick.get("final_prob"),
                "raw_probability": pick.get("raw_final_prob"),
                "confidence_evidence": pick.get("confidence_evidence"),
                "metric_card": pick.get("metric_card"),
            }
            for pick in top_2_batter_picks
        ],
    }
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(report), handle, indent=2, allow_nan=False)
        print(f"Metric report written to {report_path}")
        return report_path
    except Exception as exc:
        print(f"WARNING: could not write metric report: {safe_error(exc)}")
        return None


written_report_path = write_prediction_report()

if SPEAK_OUTPUT:
    if written_report_path:
        try:
            from speech_output import speak_report_file
            speech_backend = speak_report_file(
                written_report_path,
                style=SPEAK_STYLE,
                backend=SPEAK_BACKEND,
                voice=os.getenv("PREDICTOR_SPEAK_VOICE") or None,
                rate=int(os.getenv("PREDICTOR_SPEAK_RATE", "180")),
                print_text=True,
            )
            print(f"Spoken briefing complete using backend: {speech_backend}")
        except Exception as exc:
            print(f"WARNING: spoken briefing failed: {safe_error(exc)}")
    else:
        print("WARNING: skipped spoken briefing because the report was not written.")

print("DONE")
line()