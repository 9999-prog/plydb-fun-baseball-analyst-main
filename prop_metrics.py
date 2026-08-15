import math
from datetime import timedelta
import numpy as np
import pandas as pd

from modern_stats import blend_rate, fetch_modern_team_stats
from advanced_metrics import team_name_matches


DEFAULT_FIRST_INNING_RUNS = 0.45
DEFAULT_TEAM_RUNS = 4.2
TARGET_COVERAGE_GAMES = 10


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
    v = safe_num(value)
    return math.isfinite(v)


def clamp(value, low=0.0, high=1.0):
    v = safe_num(value)
    if not valid(v):
        return np.nan
    return max(low, min(high, v))


def pct(value):
    value = safe_num(value)
    if not valid(value):
        return "N/A"
    return f"{value * 100:.1f}%"



def _date_series(pa_df):
    """Return a comparable, timezone-naive date series without mutating input."""
    dates = pd.to_datetime(pa_df["game_date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates


def _coverage_weight(*game_counts, target=TARGET_COVERAGE_GAMES):
    """Convert the smallest observed sample into a 0-1 confidence weight."""
    counts = [int(count) for count in game_counts if count is not None]
    if not counts:
        return 0.0
    return clamp(min(counts) / float(target), 0.0, 1.0)


def _blend_with_neutral(probability, coverage):
    """Shrink a signal toward 50/50 when the historical sample is small."""
    probability = safe_num(probability)
    coverage = clamp(coverage)
    if not valid(probability) or not valid(coverage):
        return np.nan
    return 0.5 + (probability - 0.5) * coverage


def fair_american_odds(probability):
    p = safe_num(probability)
    p = clamp(p, 1e-6, 0.999999)
    if p >= 0.5:
        return int(round((p / (1.0 - p)) * 100.0))
    return int(round(-((1.0 - p) / p) * 100.0))


def _american_implied_probability(odds):
    odds = safe_num(odds)
    if not valid(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def best_total_market(home_team, away_team, odds_data, side="over"):
    """Return a robust best quote for one side of the main totals market.

    Totals are vulnerable to mixing alternate lines and stale bookmaker
    quotes. Prefer the consensus point and discard a quote whose implied
    probability is more than ten percentage points from the side's median.
    """
    candidates = []
    target = str(side).lower()
    for game in odds_data or []:
        if not ((team_name_matches(game.get("home_team"), home_team)
                 and team_name_matches(game.get("away_team"), away_team))
                or (team_name_matches(game.get("home_team"), away_team)
                    and team_name_matches(game.get("away_team"), home_team))):
            continue
        for bookmaker in game.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    if str(outcome.get("name", "")).lower() != target:
                        continue
                    point = safe_num(outcome.get("point"))
                    price = safe_num(outcome.get("price"))
                    implied = _american_implied_probability(price)
                    if not valid(point) or not valid(price) or not valid(implied):
                        continue
                    candidates.append({
                        "point": float(point),
                        "price": float(price),
                        "implied_probability": float(implied),
                    })
    if not candidates:
        return None

    consensus_point = float(np.median([row["point"] for row in candidates]))
    same_line = [
        row for row in candidates
        if abs(row["point"] - consensus_point) <= 0.01
    ]
    if not same_line:
        same_line = [min(
            candidates,
            key=lambda row: abs(row["point"] - consensus_point),
        )]

    consensus_probability = float(np.median([
        row["implied_probability"] for row in same_line
    ]))
    robust = [
        row for row in same_line
        if abs(row["implied_probability"] - consensus_probability) <= 0.10
    ]
    if not robust:
        robust = [min(
            same_line,
            key=lambda row: abs(row["implied_probability"] - consensus_probability),
        )]
    # At the same line, the largest American number is the best payout.
    return max(robust, key=lambda row: row["price"])


def team_first_inning_runs(team, pa_df, as_of_date, lookback_days=30):
    return _team_first_inning_profile(
        team, pa_df, as_of_date, lookback_days=lookback_days
    )["mean_runs"]


def _team_first_inning_profile(team, pa_df, as_of_date, lookback_days=30):
    """Return first-inning scoring and the number of games supporting it."""
    required = {"batting_team", "inning", "game_date", "game_pk", "runs_on_pa"}
    if pa_df is None or pa_df.empty or not required.issubset(pa_df.columns):
        return {"mean_runs": DEFAULT_FIRST_INNING_RUNS, "games": 0}

    ts = pd.Timestamp(as_of_date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    dates = _date_series(pa_df)
    sub = pa_df[
        (pa_df["batting_team"].astype(str).str.upper() == str(team).upper())
        & (pd.to_numeric(pa_df["inning"], errors="coerce") == 1)
        & (dates >= ts - timedelta(days=int(lookback_days)))
        & (dates < ts + timedelta(days=1))
    ].copy()
    if sub.empty:
        return {"mean_runs": DEFAULT_FIRST_INNING_RUNS, "games": 0}

    sub["runs_on_pa"] = pd.to_numeric(sub["runs_on_pa"], errors="coerce").fillna(0.0)
    runs = sub.groupby("game_pk", as_index=False)["runs_on_pa"].sum()
    if runs.empty:
        return {"mean_runs": DEFAULT_FIRST_INNING_RUNS, "games": 0}
    return {
        "mean_runs": float(runs["runs_on_pa"].mean()),
        "games": int(len(runs)),
    }


def recent_team_scoring_profile(team, pa_df, as_of_date, lookback_days=30):
    defaults = {
        "scored": DEFAULT_TEAM_RUNS,
        "allowed": DEFAULT_TEAM_RUNS,
        "scored_games": 0,
        "allowed_games": 0,
    }
    required = {
        "batting_team", "pitching_team", "game_date", "game_pk", "runs_on_pa"
    }
    if pa_df is None or pa_df.empty or not required.issubset(pa_df.columns):
        return defaults

    ts = pd.Timestamp(as_of_date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    dates = _date_series(pa_df)
    window = (dates >= ts - timedelta(days=int(lookback_days))) & (
        dates < ts + timedelta(days=1)
    )
    team_upper = str(team).upper()
    bat = pa_df[
        window
        & pa_df["batting_team"].astype(str).str.upper().eq(team_upper)
    ].copy()
    pit = pa_df[
        window
        & pa_df["pitching_team"].astype(str).str.upper().eq(team_upper)
    ].copy()

    def per_game_average(frame):
        if frame.empty:
            return DEFAULT_TEAM_RUNS, 0
        frame["runs_on_pa"] = pd.to_numeric(
            frame["runs_on_pa"], errors="coerce"
        ).fillna(0.0)
        game_runs = frame.groupby("game_pk", as_index=False)["runs_on_pa"].sum()
        if game_runs.empty:
            return DEFAULT_TEAM_RUNS, 0
        return float(game_runs["runs_on_pa"].mean()), int(len(game_runs))

    scored, scored_games = per_game_average(bat)
    allowed, allowed_games = per_game_average(pit)
    return {
        "scored": max(0.0, min(8.0, scored)),
        "allowed": max(0.0, min(8.0, allowed)),
        "scored_games": scored_games,
        "allowed_games": allowed_games,
    }


def _projected_total_profile(
    home_team, away_team, pa_df, as_of_date, modern_stats=None
):
    """Build a score/total estimate from historical and current-season rates.

    The previous implementation added a team's scoring rate to half of the
    opponent's allowed rate and then divided the *combined* score by two.
    That made a neutral 4.2/4.2 matchup project to 6.3 total runs before the
    final clamp, which was not a total-runs projection at all.  Keep offense
    and prevention on the same scale instead: each team's estimate is the
    average of its own scoring rate and the opponent's allowed rate, and the
    game total is the sum of those two score estimates.
    """
    if modern_stats is None:
        modern_stats = fetch_modern_team_stats(pd.Timestamp(as_of_date).year)

    historical_home = recent_team_scoring_profile(home_team, pa_df, as_of_date)
    historical_away = recent_team_scoring_profile(away_team, pa_df, as_of_date)
    modern_home = modern_stats.get(str(home_team).upper(), {})
    modern_away = modern_stats.get(str(away_team).upper(), {})

    def component(historical, historical_games, modern, modern_field):
        modern_value = modern.get(modern_field)
        value, effective_modern_weight = blend_rate(
            historical,
            historical_games,
            modern_value,
            modern.get("modern_games", 0),
        )
        modern_support = (
            int(modern.get("modern_games", 0))
            if valid(modern_value)
            else 0
        )
        return (
            historical if value is None else value,
            max(int(historical_games), modern_support),
            effective_modern_weight,
        )

    home_scored, home_scored_games, home_scored_weight = component(
        historical_home["scored"],
        historical_home["scored_games"],
        modern_home,
        "runs_per_game",
    )
    home_allowed, home_allowed_games, home_allowed_weight = component(
        historical_home["allowed"],
        historical_home["allowed_games"],
        modern_home,
        "runs_allowed_per_game",
    )
    away_scored, away_scored_games, away_scored_weight = component(
        historical_away["scored"],
        historical_away["scored_games"],
        modern_away,
        "runs_per_game",
    )
    away_allowed, away_allowed_games, away_allowed_weight = component(
        historical_away["allowed"],
        historical_away["allowed_games"],
        modern_away,
        "runs_allowed_per_game",
    )

    proj_home = 0.5 * (home_scored + away_allowed)
    proj_away = 0.5 * (away_scored + home_allowed)
    projected_total = clamp(proj_home + proj_away, 4.0, 12.0)
    support_games = min(
        home_scored_games,
        home_allowed_games,
        away_scored_games,
        away_allowed_games,
    )
    coverage = _coverage_weight(
        home_scored_games,
        home_allowed_games,
        away_scored_games,
        away_allowed_games,
    )
    return {
        "projected_home_runs": clamp(proj_home, 0.0, 10.0),
        "projected_away_runs": clamp(proj_away, 0.0, 10.0),
        "projected_total": projected_total,
        "coverage": coverage,
        "sample_games": support_games,
        "modern_weight": min(
            home_scored_weight,
            home_allowed_weight,
            away_scored_weight,
            away_allowed_weight,
        ),
    }


def projected_score_profile(
    home_team, away_team, pa_df, as_of_date, modern_stats=None
):
    """Return projected team scores plus support metadata.

    This is intentionally a small, transparent run-rate model.  It is useful
    for the metric cards and explanations, but it is not presented as a
    full game simulator: it does not fabricate weather, confirmed lineups,
    or inning-level distributions that are not present in the data.
    """
    profile = _projected_total_profile(
        home_team, away_team, pa_df, as_of_date, modern_stats=modern_stats
    )
    profile["projected_margin_home"] = (
        profile["projected_home_runs"] - profile["projected_away_runs"]
    )
    return profile


def projected_total_runs(home_team, away_team, pa_df, as_of_date, modern_stats=None):
    return projected_score_profile(
        home_team, away_team, pa_df, as_of_date, modern_stats=modern_stats
    )["projected_total"]


def projected_nrfi_prob(home_team, away_team, pa_df, as_of_date):
    home = _team_first_inning_profile(home_team, pa_df, as_of_date)
    away = _team_first_inning_profile(away_team, pa_df, as_of_date)
    total_runs = max(0.1, home["mean_runs"] + away["mean_runs"])
    raw_probability = clamp(math.exp(-total_runs * 0.9), 0.15, 0.9)
    coverage = _coverage_weight(home["games"], away["games"])
    return clamp(_blend_with_neutral(raw_probability, coverage), 0.15, 0.9)


def projected_rifi_prob(home_team, away_team, pa_df, as_of_date):
    return 1.0 - projected_nrfi_prob(home_team, away_team, pa_df, as_of_date)


def totals_pick(
    home_team,
    away_team,
    pa_df,
    as_of_date,
    odds_data,
    modern_stats=None,
    score_profile=None,
):
    # The team scorer already builds this profile. Reusing it avoids another
    # full scan of the local PA table for every printed section.
    if isinstance(score_profile, dict) and valid(score_profile.get("projected_total")):
        total_profile = score_profile
    else:
        total_profile = _projected_total_profile(
            home_team, away_team, pa_df, as_of_date, modern_stats=modern_stats
        )
    total_proj = total_profile["projected_total"]
    over_line = best_total_market(home_team, away_team, odds_data, side="over")
    under_line = best_total_market(home_team, away_team, odds_data, side="under")
    # If we have an actual market line use it; otherwise leave market_line None.
    if over_line:
        line = float(over_line["point"])
    elif under_line:
        line = float(under_line["point"])
    else:
        line = None

    # Never turn a stale/default run average into a confident prop.  The raw
    # logistic estimate is shrunk toward 50/50 using the smallest team sample.
    if line is None:
        over_prob = 0.5
    else:
        raw_over_prob = 1.0 / (1.0 + math.exp(-(total_proj - line) * 0.9))
        over_prob = _blend_with_neutral(
            raw_over_prob, total_profile["coverage"]
        )
    under_prob = 1.0 - over_prob
    pick = (
        "PASS"
        if (
            line is None
            or total_profile["coverage"] < 0.5
            or abs(over_prob - 0.5) < 0.03
        )
        else ("over" if over_prob > 0.5 else "under")
    )
    return {
        "home_team": home_team,
        "away_team": away_team,
        "projected_total": total_proj,
        "market_line": line,
        "model_over_prob": over_prob,
        "model_under_prob": under_prob,
        "best_over_price": over_line["price"] if over_line else None,
        "best_under_price": under_line["price"] if under_line else None,
        "data_coverage": total_profile["coverage"],
        "sample_games": total_profile["sample_games"],
        "modern_weight": total_profile["modern_weight"],
        "pick": pick,
    }


def score_numerical_prop(home_team, away_team, pa_df, as_of_date, odds_data, modern_stats=None):
    under = totals_pick(
        home_team, away_team, pa_df, as_of_date, odds_data, modern_stats=modern_stats
    )
    nrfi_probability = projected_nrfi_prob(home_team, away_team, pa_df, as_of_date)
    nrfi = {
        "home_team": home_team,
        "away_team": away_team,
        "prob": nrfi_probability,
        "rifi_prob": 1.0 - nrfi_probability,
        "pick": "NRFI",
        "fair_price": fair_american_odds(nrfi_probability),
    }
    return {"under": under, "nrfi": nrfi}


def select_best_under(games, pa_df, as_of_date, odds_data, modern_stats=None):
    candidate = None
    for game in games:
        under = totals_pick(
            game["home_team"], game["away_team"], pa_df, as_of_date, odds_data,
            modern_stats=modern_stats,
        )
        if candidate is None or under["model_under_prob"] > candidate["model_under_prob"]:
            candidate = under
    return candidate


def select_best_nrfi(games, pa_df, as_of_date):
    candidate = None
    for game in games:
        prob = projected_nrfi_prob(game["home_team"], game["away_team"], pa_df, as_of_date)
        if candidate is None or prob > candidate["prob"]:
            candidate = {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "prob": prob,
                "fair_price": fair_american_odds(prob),
            }
    return candidate


def metric_small_block(under_pick, nrfi_pick):
    under_label = "U" if under_pick else "U N/A"
    nrfi_label = "NRFI" if nrfi_pick else "NRFI N/A"
    if under_pick:
        ml = under_pick.get('market_line')
        ml_str = f"{ml:.1f}" if valid(ml) else "N/A"
        under_label = f"U {ml_str} {pct(under_pick['model_under_prob'])}"
    if nrfi_pick:
        nrfi_label = f"NRFI {pct(nrfi_pick['prob'])}"
    return f"\033[2m{under_label} | {nrfi_label}\033[0m"


def prop_reason_under(under_pick):
    """Explain the total direction without assuming every pick is an under."""
    if not under_pick:
        return "The total model is not finding an actionable edge tonight."
    proj = safe_num(under_pick.get("projected_total"))
    prob_under = safe_num(under_pick.get("model_under_prob"))
    prob_over = safe_num(under_pick.get("model_over_prob"))
    ml = safe_num(under_pick.get("market_line"))
    coverage = safe_num(under_pick.get("data_coverage"))
    sample_games = under_pick.get("sample_games", 0)
    if valid(coverage) and coverage < 0.5:
        return (
            f"Only {sample_games} recent team-games support the total estimate, "
            "so the model is shrinking this prop toward neutral instead of treating "
            "a stale box-score average as a reliable edge."
        )
    if not valid(ml):
        return (
            f"The model projects {proj:.1f} total runs, but there is no market total "
            "available to compare, so it is not issuing a priced over/under pick."
        )
    if valid(prob_under) and abs(prob_under - 0.5) < 0.03:
        return (
            f"The projection is {proj:.1f} runs against a {ml:.1f} market line, "
            "but the difference is too small for an actionable total edge."
        )
    if valid(prob_over) and prob_over > 0.5:
        return (
            f"The model projects {proj:.1f} total runs, above the market line of "
            f"{ml:.1f}, supporting the over with a {pct(prob_over)} model chance."
        )
    return (
        f"The model projects {proj:.1f} total runs, below the market line of "
        f"{ml:.1f}, supporting the under with a {pct(prob_under)} model chance."
    )


def prop_reason_nrfi(nrfi_pick):
    if not nrfi_pick:
        return "The first-inning environment is too messy for a clean NRFI edge."
    probability = safe_num(nrfi_pick.get("prob"))
    if valid(probability) and abs(probability - 0.5) < 0.03:
        return (
            "The model is neutral here: there is not enough recent first-inning "
            "sample to distinguish this matchup from a 50/50 NRFI/RIFI outcome."
        )
    return (
        f"The observed recent first-inning run environment suggests a {pct(nrfi_pick['prob'])} "
        "NRFI chance; this is a rate-based signal, not a guarantee about a confirmed lineup or starter."
    )
