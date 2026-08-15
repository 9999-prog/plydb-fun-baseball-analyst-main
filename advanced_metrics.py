"""Transparent metric cards for the MLB matchup analyst.

The screenshot supplied with the project contains useful *categories* of
metrics, but it does not define the formulas or provide trustworthy numeric
values. This module implements the defensible parts from real local Statcast,
MLB Stats API context, and the market feed:

* expected value and vig-adjusted market edge;
* confidence-adjusted AI probability (explicitly not claimed to be a fitted
  post-hoc calibration model);
* base projection, projected score, total, and margin;
* last-5/last-10 form, prior-10 deltas, usage, impact, and uncertainty;
* directional team H2H and batter/pitcher H2H with sample-size shrinkage.

Metrics whose screenshot names are not standard or whose definitions cannot be
recovered from the data (for example ``Sponge Coeff``) are returned as
``None`` with an ``UNDEFINED`` status rather than being invented. Every
returned card includes sample counts and source/quality metadata so a small or
stale sample cannot look like a precise estimate.
"""

from __future__ import annotations

from datetime import timedelta
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_PA_HIT_RATE = 0.24
DEFAULT_GAME_HIT_RATE = 0.57
DEFAULT_GAME_WIN_RATE = 0.50
H2H_SHRINK_STRENGTH = 6.0
FORM_SHRINK_STRENGTH = 5.0
MIN_H2H_GAMES = 3
MIN_BVP_PA = 3
# A single bookmaker can occasionally return a stale/malformed price (for
# example +2500 in an otherwise ordinary two-way market). Keep the best
# plausible price, but do not let that quote create a fake EV edge. Ten
# percentage points is deliberately generous for cross-book moneylines.
MARKET_OUTLIER_PROBABILITY_TOLERANCE = 0.10


def finite(value: Any, default: float | None = None) -> float | None:
    """Return a finite float or ``default`` without leaking NaN to JSON."""
    try:
        if value is None or pd.isna(value):
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float | None:
    value = finite(value)
    if value is None:
        return None
    return max(low, min(high, value))


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _dates(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates


def shrink_rate(
    observed_rate: Any,
    sample_size: Any,
    prior_rate: float,
    prior_strength: float,
) -> float | None:
    """Empirical-Bayes-style shrinkage toward a stated neutral prior."""
    observed = finite(observed_rate)
    n = finite(sample_size, 0.0) or 0.0
    prior = finite(prior_rate)
    strength = finite(prior_strength, 0.0) or 0.0
    if observed is None or prior is None or n <= 0:
        return None
    if strength <= 0:
        return observed
    return (observed * n + prior * strength) / (n + strength)


def sample_status(sample_size: Any, target: int, *, minimum: int = 1) -> str:
    n = int(finite(sample_size, 0.0) or 0)
    if n < minimum:
        return "NO_DATA"
    if n < target:
        return "LOW_SAMPLE"
    return "SUPPORTED"


def confidence_adjusted_probability(
    raw_probability: Any,
    *,
    signal_quality: Any = 0.5,
    modern_games: Any = 0,
    local_staleness_days: Any = None,
    modern_target_games: int = 10,
) -> tuple[float | None, float]:
    """Shrink an estimate toward 50% when evidence is weak or stale.

    This is a confidence adjustment, not a learned calibration curve. A
    genuine calibration model needs out-of-sample predictions and is not
    available in the existing artifact. Naming the method explicitly avoids
    presenting a hand-built shrinkage rule as if it had been fit on outcomes.
    """
    raw = clamp(raw_probability)
    if raw is None:
        return None, 0.0
    quality = clamp(signal_quality)
    quality = 0.5 if quality is None else quality
    modern = clamp(
        (finite(modern_games, 0.0) or 0.0) / max(1.0, float(modern_target_games))
    )
    modern = 0.0 if modern is None else modern
    evidence = 0.35 + 0.35 * quality + 0.30 * modern
    stale = finite(local_staleness_days)
    if stale is not None and stale > 14:
        evidence *= max(0.35, 1.0 - min(0.60, (stale - 14.0) / 180.0))
    evidence = max(0.20, min(0.95, evidence))
    return 0.5 + (raw - 0.5) * evidence, evidence


def american_to_probability(odds: Any) -> float | None:
    odds = finite(odds)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: Any) -> float | None:
    odds = finite(odds)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def expected_value_per_unit(probability: Any, american_odds: Any) -> float | None:
    """Expected profit per one unit staked at the supplied American price."""
    probability = clamp(probability)
    decimal = american_to_decimal(american_odds)
    if probability is None or decimal is None:
        return None
    return probability * decimal - 1.0


TEAM_ALIASES = {
    "ARI": {"ARI", "AZ", "ARIZONA DIAMONDBACKS", "DIAMONDBACKS"},
    "ATH": {"ATH", "OAK", "ATHLETICS", "OAKLAND ATHLETICS"},
    "ATL": {"ATL", "ATLANTA BRAVES"},
    "BAL": {"BAL", "BALTIMORE ORIOLES"},
    "BOS": {"BOS", "BOSTON RED SOX"},
    "CHC": {"CHC", "CHICAGO CUBS"},
    "CWS": {"CWS", "CHW", "CHICAGO WHITE SOX"},
    "CIN": {"CIN", "CINCINNATI REDS"},
    "CLE": {"CLE", "CLEVELAND GUARDIANS"},
    "COL": {"COL", "COLORADO ROCKIES"},
    "DET": {"DET", "DETROIT TIGERS"},
    "HOU": {"HOU", "HOUSTON ASTROS"},
    "KC": {"KC", "KCR", "KANSAS CITY ROYALS"},
    "LAA": {"LAA", "LOS ANGELES ANGELS", "ANAHEIM ANGELS"},
    "LAD": {"LAD", "LOS ANGELES DODGERS"},
    "MIA": {"MIA", "MIAMI MARLINS"},
    "MIL": {"MIL", "MILWAUKEE BREWERS"},
    "MIN": {"MIN", "MINNESOTA TWINS"},
    "NYM": {"NYM", "NEW YORK METS"},
    "NYY": {"NYY", "NEW YORK YANKEES"},
    "PHI": {"PHI", "PHILADELPHIA PHILLIES"},
    "PIT": {"PIT", "PITTSBURGH PIRATES"},
    "SD": {"SD", "SDP", "SAN DIEGO PADRES"},
    "SEA": {"SEA", "SEATTLE MARINERS"},
    "SF": {"SF", "SFG", "SAN FRANCISCO GIANTS"},
    "STL": {"STL", "ST. LOUIS CARDINALS", "ST LOUIS CARDINALS"},
    "TB": {"TB", "TBR", "TAMPA BAY RAYS"},
    "TEX": {"TEX", "TEXAS RANGERS"},
    "TOR": {"TOR", "TORONTO BLUE JAYS"},
    "WSH": {"WSH", "WSN", "WASHINGTON NATIONALS"},
}


def _normalise_team_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def team_name_key(value: Any) -> str:
    normalised = _normalise_team_name(value)
    for canonical, aliases in TEAM_ALIASES.items():
        if normalised in {_normalise_team_name(alias) for alias in aliases}:
            return canonical
    return normalised


def team_name_matches(left: Any, right: Any) -> bool:
    return bool(team_name_key(left)) and team_name_key(left) == team_name_key(right)


def _market_game(odds_data: Any, home_team: str, away_team: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for game in odds_data or []:
        market_home = str(game.get("home_team", ""))
        market_away = str(game.get("away_team", ""))
        if not ((team_name_matches(market_home, home_team)
                 and team_name_matches(market_away, away_team))
                or (team_name_matches(market_home, away_team)
                    and team_name_matches(market_away, home_team))):
            continue
        for bookmaker in game.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", []) or []
                home_price = next(
                    (finite(outcome.get("price")) for outcome in outcomes
                     if team_name_matches(outcome.get("name"), home_team)),
                    None,
                )
                away_price = next(
                    (finite(outcome.get("price")) for outcome in outcomes
                     if team_name_matches(outcome.get("name"), away_team)),
                    None,
                )
                if home_price is None or away_price is None:
                    continue
                home_raw = american_to_probability(home_price)
                away_raw = american_to_probability(away_price)
                if home_raw is None or away_raw is None or home_raw + away_raw <= 0:
                    continue
                records.append({
                    "home_price": home_price,
                    "away_price": away_price,
                    "home_raw_probability": home_raw,
                    "away_raw_probability": away_raw,
                    "home_no_vig_probability": home_raw / (home_raw + away_raw),
                    "away_no_vig_probability": away_raw / (home_raw + away_raw),
                })
    return records


def market_card(odds_data: Any, home_team: str, away_team: str) -> dict[str, Any]:
    """Use a consensus probability and a robust best available price.

    The median no-vig probability is more stable than one book's implied
    probability. Best-price selection is restricted to quotes within a
    generous distance of that consensus so an obvious stale/outlier quote
    cannot manufacture a huge EV number. If every quote is unusual, the
    closest quote is retained and the card exposes the filter status.
    """
    records = _market_game(odds_data, home_team, away_team)
    if not records:
        return {
            "available": False,
            "home_market_probability": None,
            "away_market_probability": None,
            "home_best_price": None,
            "away_best_price": None,
            "book_count": 0,
            "robust_book_count": 0,
            "outlier_count": 0,
            "price_filter": "NO_MARKET",
        }

    home_market_probability = float(np.median([
        r["home_no_vig_probability"] for r in records
    ]))
    away_market_probability = float(np.median([
        r["away_no_vig_probability"] for r in records
    ]))

    def robust_records(side: str, consensus: float) -> list[dict[str, Any]]:
        probability_key = f"{side}_no_vig_probability"
        eligible = [
            record for record in records
            if abs(record[probability_key] - consensus)
            <= MARKET_OUTLIER_PROBABILITY_TOLERANCE
        ]
        if eligible:
            return eligible
        # A one-book market should remain usable. Choose the nearest quote
        # instead of silently discarding the market altogether.
        return [min(
            records,
            key=lambda record: abs(record[probability_key] - consensus),
        )]

    home_eligible = robust_records("home", home_market_probability)
    away_eligible = robust_records("away", away_market_probability)
    robust_records_used = {id(record) for record in home_eligible + away_eligible}
    return {
        "available": True,
        "home_market_probability": home_market_probability,
        "away_market_probability": away_market_probability,
        "home_best_price": max(r["home_price"] for r in home_eligible),
        "away_best_price": max(r["away_price"] for r in away_eligible),
        "book_count": len(records),
        "robust_book_count": len(robust_records_used),
        "outlier_count": max(0, len(records) - len(robust_records_used)),
        "price_filter": (
            "ROBUST"
            if len(robust_records_used) < len(records)
            else "ALL_QUOTES_WITHIN_TOLERANCE"
        ),
    }


def _hit_column(frame: pd.DataFrame) -> pd.Series:
    if "is_hit" in frame.columns:
        return pd.to_numeric(frame["is_hit"], errors="coerce").fillna(0).clip(0, 1)
    events = frame.get("events", pd.Series("", index=frame.index))
    return events.fillna("").astype(str).str.lower().isin(
        {"single", "double", "triple", "home_run"}
    ).astype(float)


def _window_game_ids(
    frame: pd.DataFrame,
    n_games: int,
    *,
    excluded_game_ids: set[Any] | None = None,
) -> list[Any]:
    if frame.empty or "game_pk" not in frame.columns:
        return []
    excluded_game_ids = excluded_game_ids or set()
    games = (
        frame[["game_pk", "game_date"]]
        .dropna(subset=["game_pk", "game_date"])
        .drop_duplicates("game_pk")
        .sort_values(["game_date", "game_pk"], ascending=[False, False])
    )
    games = games[~games["game_pk"].isin(excluded_game_ids)].head(int(n_games))
    return games["game_pk"].tolist()


def _game_window_summary(frame: pd.DataFrame, game_ids: list[Any]) -> dict[str, Any]:
    if not game_ids:
        return {
            "games": 0, "pa": 0, "hits": 0, "hit_games": 0,
            "raw_game_hit_rate": None, "game_hit_rate": None,
            "pa_hit_rate": None, "xba": None, "xwoba": None,
            "hard_hit_rate": None, "hr_rate": None,
        }
    sub = frame[frame["game_pk"].isin(game_ids)].copy()
    sub["_is_hit"] = _hit_column(sub)
    group = sub.groupby("game_pk", dropna=False)
    hit_games = group["_is_hit"].max()

    def numeric_mean(column: str) -> float | None:
        if column not in sub.columns:
            return None
        values = pd.to_numeric(sub[column], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    pa = int(len(sub))
    hits = int(sub["_is_hit"].sum())
    games = int(len(hit_games))
    hard_hit = None
    if "launch_speed" in sub.columns:
        speed = pd.to_numeric(sub["launch_speed"], errors="coerce")
        if speed.notna().any():
            hard_hit = float(speed.ge(95).mean())
    home_runs = None
    events = sub.get("events")
    if events is not None:
        home_runs = int(events.fillna("").astype(str).str.lower().eq("home_run").sum())
    raw_rate = float(hit_games.mean()) if games else None
    return {
        "games": games, "pa": pa, "hits": hits,
        "hit_games": int(hit_games.sum()),
        "raw_game_hit_rate": raw_rate,
        "game_hit_rate": shrink_rate(raw_rate, games, DEFAULT_GAME_HIT_RATE, FORM_SHRINK_STRENGTH),
        "pa_hit_rate": hits / pa if pa else None,
        "xba": numeric_mean("estimated_ba_using_speedangle"),
        "xwoba": numeric_mean("estimated_woba_using_speedangle"),
        "hard_hit_rate": hard_hit,
        "hr_rate": home_runs / pa if home_runs is not None and pa else None,
    }


def league_reference_context(
    pa_df: pd.DataFrame,
    as_of_date: Any,
    *,
    lookback_days: int = 365,
) -> dict[str, float | None]:
    """Compute reference rates from the supplied real PA data only."""
    if pa_df is None or pa_df.empty or "game_date" not in pa_df.columns:
        return {"xba": None, "xwoba": None, "hard_hit_rate": None}
    ts = _timestamp(as_of_date)
    dates = _dates(pa_df)
    frame = pa_df[(dates < ts) & (dates >= ts - timedelta(days=lookback_days))]
    if frame.empty:
        frame = pa_df[dates < ts]
    result: dict[str, float | None] = {}
    for source, key in [
        ("estimated_ba_using_speedangle", "xba"),
        ("estimated_woba_using_speedangle", "xwoba"),
    ]:
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce").dropna()
            result[key] = float(values.mean()) if not values.empty else None
        else:
            result[key] = None
    if "launch_speed" in frame.columns:
        speed = pd.to_numeric(frame["launch_speed"], errors="coerce")
        result["hard_hit_rate"] = float(speed.ge(95).mean()) if speed.notna().any() else None
    else:
        result["hard_hit_rate"] = None
    return result


def _impact_rating(summary: Mapping[str, Any], context: Mapping[str, Any]) -> float | None:
    specs = [("xwoba", 0.50, 0.08), ("xba", 0.30, 0.06), ("hard_hit_rate", 0.20, 0.15)]
    values = []
    total_weight = 0.0
    for key, weight, scale in specs:
        observed = finite(summary.get(key))
        reference = finite(context.get(key))
        if observed is None or reference is None:
            continue
        relative = max(-2.0, min(2.0, (observed - reference) / scale))
        values.append((50.0 + relative * 25.0) * weight)
        total_weight += weight
    return sum(values) / total_weight if total_weight else None


def _quality_delta(recent: Mapping[str, Any], prior: Mapping[str, Any]) -> float | None:
    for key in ("xwoba", "xba", "pa_hit_rate", "hard_hit_rate"):
        recent_value = finite(recent.get(key))
        prior_value = finite(prior.get(key))
        if recent_value is not None and prior_value is not None:
            return recent_value - prior_value
    return None


def _wilson_lower(successes: Any, trials: Any, z: float = 1.96) -> float | None:
    successes = finite(successes)
    trials = finite(trials)
    if successes is None or trials is None or trials <= 0:
        return None
    successes = max(0.0, min(trials, successes))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return max(0.0, (centre - spread) / denominator)

def batter_form_metrics(
    pa_df: pd.DataFrame,
    batter_id: Any,
    as_of_date: Any,
    *,
    reference_context: Mapping[str, Any] | None = None,
    local_staleness_days: Any = None,
) -> dict[str, Any]:
    """Build leakage-safe batter form cards from completed games only."""
    empty = {
        "batter_id": finite(batter_id),
        "last5_games": 0, "last10_games": 0,
        "last5_hit_rate": None, "last10_hit_rate": None,
        "last5_pa_hit_rate": None, "last10_pa_hit_rate": None,
        "last5_form_delta": None, "last10_vs_prior10_form_delta": None,
        "usage_pa_per_game": None, "usage_rating": None,
        "impact_rating": None, "minimum_credible_hit_rate": None,
        "efficiency_dropoff": None, "data_status": "NO_DATA",
    }
    required = {"batter", "game_pk", "game_date"}
    if pa_df is None or pa_df.empty or not required.issubset(pa_df.columns):
        return empty
    ts = _timestamp(as_of_date)
    dates = _dates(pa_df)
    batter = pa_df[
        (pd.to_numeric(pa_df["batter"], errors="coerce") == finite(batter_id))
        & (dates < ts)
    ].copy()
    if batter.empty:
        return empty
    batter["game_date"] = _dates(batter)
    last5_ids = _window_game_ids(batter, 5)
    last10_ids = _window_game_ids(batter, 10)
    prior10_ids = _window_game_ids(batter, 10, excluded_game_ids=set(last10_ids))
    last5 = _game_window_summary(batter, last5_ids)
    last10 = _game_window_summary(batter, last10_ids)
    prior10 = _game_window_summary(batter, prior10_ids)
    context = reference_context or league_reference_context(pa_df, as_of_date)
    usage_games = last10["games"]
    usage_pa = last10["pa"]
    usage_pa_per_game = usage_pa / usage_games if usage_games else None
    usage_rating = (
        100.0 * min(1.0, usage_pa_per_game / 4.2)
        if usage_pa_per_game is not None else None
    )
    result = dict(empty)
    result.update({
        "last5_games": last5["games"], "last10_games": last10["games"],
        "last5_pa": last5["pa"], "last10_pa": last10["pa"],
        "last5_hit_rate_raw": last5["raw_game_hit_rate"],
        "last10_hit_rate_raw": last10["raw_game_hit_rate"],
        "last5_hit_rate": last5["game_hit_rate"],
        "last10_hit_rate": last10["game_hit_rate"],
        "last5_pa_hit_rate": last5["pa_hit_rate"],
        "last10_pa_hit_rate": last10["pa_hit_rate"],
        "last5_xba": last5["xba"], "last10_xba": last10["xba"],
        "last10_xwoba": last10["xwoba"],
        "last10_hard_hit_rate": last10["hard_hit_rate"],
        "last10_hr_rate": last10["hr_rate"],
        "last5_form_delta": (
            last5["game_hit_rate"] - last10["game_hit_rate"]
            if finite(last5["game_hit_rate"]) is not None
            and finite(last10["game_hit_rate"]) is not None else None
        ),
        "last10_vs_prior10_form_delta": (
            last10["game_hit_rate"] - prior10["game_hit_rate"]
            if finite(last10["game_hit_rate"]) is not None
            and finite(prior10["game_hit_rate"]) is not None else None
        ),
        "usage_pa_per_game": usage_pa_per_game,
        "usage_rating": usage_rating,
        "impact_rating": _impact_rating(last10, context),
        "minimum_credible_hit_rate": _wilson_lower(
            last10.get("hit_games"), last10.get("games")
        ),
        "efficiency_dropoff": _quality_delta(last10, prior10),
        "data_status": sample_status(last10["games"], 10),
        "data_staleness_days": finite(local_staleness_days),
        "reference_context": dict(context),
    })
    return result


def _team_game_results(
    pa_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: Any,
) -> pd.DataFrame:
    required = {"game_pk", "game_date", "home_team", "away_team"}
    if pa_df is None or pa_df.empty or not required.issubset(pa_df.columns):
        return pd.DataFrame()
    ts = _timestamp(as_of_date)
    dates = _dates(pa_df)
    home_mask = pa_df["home_team"].astype(str).str.upper()
    away_mask = pa_df["away_team"].astype(str).str.upper()
    h = str(home_team).upper()
    a = str(away_team).upper()
    sub = pa_df[
        (dates < ts)
        & (((home_mask == h) & (away_mask == a)) | ((home_mask == a) & (away_mask == h)))
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    if "batting_team" not in sub.columns:
        top = sub.get("inning_topbot", pd.Series("", index=sub.index)).astype(str).str.lower().eq("top")
        sub["batting_team"] = np.where(top, sub["away_team"], sub["home_team"])
    if "runs_on_pa" not in sub.columns:
        if "bat_score" in sub.columns and "post_bat_score" in sub.columns:
            sub["runs_on_pa"] = (
                pd.to_numeric(sub["post_bat_score"], errors="coerce")
                - pd.to_numeric(sub["bat_score"], errors="coerce")
            ).clip(lower=0).fillna(0.0)
        else:
            sub["runs_on_pa"] = 0.0
    rows: list[dict[str, Any]] = []
    for game_pk, group in sub.groupby("game_pk", sort=False):
        group = group.sort_values("game_date")
        actual_home = str(group["home_team"].iloc[0]).upper()
        actual_away = str(group["away_team"].iloc[0]).upper()
        home_score = away_score = None
        if "post_home_score" in group.columns and "post_away_score" in group.columns:
            hs = pd.to_numeric(group["post_home_score"], errors="coerce").dropna()
            aws = pd.to_numeric(group["post_away_score"], errors="coerce").dropna()
            if not hs.empty and not aws.empty:
                home_score, away_score = float(hs.iloc[-1]), float(aws.iloc[-1])
        if home_score is None or away_score is None:
            runs = pd.to_numeric(group["runs_on_pa"], errors="coerce").fillna(0.0)
            temp = group.assign(_runs=runs)
            scored = temp.groupby(temp["batting_team"].astype(str).str.upper())["_runs"].sum()
            home_score = float(scored.get(actual_home, 0.0))
            away_score = float(scored.get(actual_away, 0.0))
        home_win = (home_score > away_score) if actual_home == h else (away_score > home_score)
        rows.append({
            "game_pk": game_pk, "game_date": pd.to_datetime(group["game_date"].iloc[0]),
            "home_score": home_score, "away_score": away_score,
            "home_team_won": bool(home_win),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["game_date", "game_pk"], ascending=[False, False])


def team_h2h_metrics(
    pa_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: Any,
) -> dict[str, Any]:
    games = _team_game_results(pa_df, home_team, away_team, as_of_date)
    if games.empty:
        return {"h2h_l5": None, "h2h_l10": None, "h2h_delta": None,
                "h2h_games": 0, "h2h_status": "NO_DATA"}
    out: dict[str, Any] = {}
    for label, count in (("h2h_l5", 5), ("h2h_l10", 10)):
        sample = games.head(count)
        raw = float(sample["home_team_won"].mean())
        out[label] = shrink_rate(raw, len(sample), DEFAULT_GAME_WIN_RATE, H2H_SHRINK_STRENGTH)
        out[f"{label}_raw"] = raw
        out[f"{label}_games"] = int(len(sample))
    preferred = out["h2h_l10"] if out["h2h_l10"] is not None else out["h2h_l5"]
    out.update({
        "h2h_delta": preferred - 0.5 if preferred is not None else None,
        "h2h_games": int(len(games)),
        "h2h_status": sample_status(len(games), 10, minimum=MIN_H2H_GAMES),
    })
    return out


def batter_pitcher_h2h_metrics(
    pa_df: pd.DataFrame,
    batter_id: Any,
    pitcher_id: Any,
    as_of_date: Any,
) -> dict[str, Any]:
    """Recent batter/pitcher game outcomes, never inferred from career totals."""
    empty = {"h2h_l5": None, "h2h_l10": None, "h2h_delta": None,
             "h2h_games": 0, "h2h_pa": 0, "h2h_status": "NO_DATA"}
    required = {"batter", "pitcher", "game_pk", "game_date"}
    if pa_df is None or pa_df.empty or not required.issubset(pa_df.columns):
        return empty
    ts = _timestamp(as_of_date)
    dates = _dates(pa_df)
    sub = pa_df[
        (pd.to_numeric(pa_df["batter"], errors="coerce") == finite(batter_id))
        & (pd.to_numeric(pa_df["pitcher"], errors="coerce") == finite(pitcher_id))
        & (dates < ts)
    ].copy()
    if sub.empty:
        return empty
    sub["_is_hit"] = _hit_column(sub)
    games = sub[["game_pk", "game_date"]].drop_duplicates("game_pk").sort_values(
        ["game_date", "game_pk"], ascending=[False, False]
    )
    rows = []
    for game_pk in games["game_pk"].tolist():
        group = sub[sub["game_pk"] == game_pk]
        rows.append({"game_pk": game_pk, "hit": int(group["_is_hit"].max())})
    game_results = pd.DataFrame(rows)
    out = dict(empty)
    out["h2h_pa"] = int(len(sub))
    out["h2h_games"] = int(len(game_results))
    for label, count in (("h2h_l5", 5), ("h2h_l10", 10)):
        sample = game_results.head(count)
        if sample.empty:
            continue
        raw = float(sample["hit"].mean())
        out[label] = shrink_rate(raw, len(sample), DEFAULT_GAME_HIT_RATE, H2H_SHRINK_STRENGTH)
        out[f"{label}_raw"] = raw
        out[f"{label}_games"] = int(len(sample))
    preferred = out["h2h_l10"] if out["h2h_l10"] is not None else out["h2h_l5"]
    out["h2h_delta"] = preferred - DEFAULT_GAME_HIT_RATE if preferred is not None else None
    out["h2h_status"] = (
        "SUPPORTED" if out["h2h_pa"] >= MIN_BVP_PA and out["h2h_games"] >= MIN_H2H_GAMES
        else "LOW_SAMPLE"
    )
    return out

def modern_batter_context(
    modern_player: Mapping[str, Any] | None,
    *,
    usage_pa_per_game: Any = None,
) -> dict[str, Any]:
    """Translate current-season PA rate into an explicitly labelled proxy.

    MLB's player-season endpoint gives hits and plate appearances, not the
    number of games in which a player had at least one hit. The optional
    ``modern_game_hit_proxy`` therefore uses observed recent PA/game and an
    independence approximation. It is context, not a replacement for the
    trained game-level model.
    """
    modern_player = modern_player or {}
    hits = finite(modern_player.get("hits"))
    at_bats = finite(modern_player.get("at_bats"))
    games = int(finite(modern_player.get("games_played"), 0.0) or 0)
    pa = finite(modern_player.get("plate_appearances"))
    if pa is None and at_bats is not None:
        pa = at_bats
    if hits is None or pa is None or pa <= 0 or games <= 0:
        return {
            "available": False, "games": games, "pa": pa or 0,
            "pa_hit_rate": None, "modern_game_hit_proxy": None,
            "modern_weight": 0.0,
        }
    pa_rate = clamp(hits / pa)
    observed_pa_per_game = finite(usage_pa_per_game)
    if observed_pa_per_game is None or observed_pa_per_game <= 0:
        observed_pa_per_game = pa / games
    game_proxy = 1.0 - (1.0 - (pa_rate or 0.0)) ** max(1.0, observed_pa_per_game)
    modern_weight = min(0.70, 0.70 * min(1.0, games / 10.0))
    return {
        "available": True, "games": games, "pa": pa,
        "pa_hit_rate": pa_rate, "modern_game_hit_proxy": clamp(game_proxy),
        "modern_weight": modern_weight,
        "proxy_note": "PA hit rate converted using observed PA/game; not a game-hit observation",
    }


def build_team_metric_cards(
    result: Mapping[str, Any],
    *,
    score_profile: Mapping[str, Any] | None,
    h2h: Mapping[str, Any] | None,
    odds_data: Any,
    local_staleness_days: Any = None,
) -> dict[str, dict[str, Any]]:
    """Build one screenshot-style card for each side of a game."""
    home = str(result.get("home_team", ""))
    away = str(result.get("away_team", ""))
    market = market_card(odds_data, home, away)
    score_profile = score_profile or {}
    h2h = h2h or {}
    cards: dict[str, dict[str, Any]] = {}
    for side, team, opponent in (("home", home, away), ("away", away, home)):
        raw = finite(result.get(f"{side}_win_prob"))
        ai, evidence = confidence_adjusted_probability(
            raw,
            signal_quality=result.get("signal_quality", 0.5),
            modern_games=result.get(f"modern_games_{side}", 0),
            local_staleness_days=local_staleness_days,
        )
        market_probability = market.get(f"{side}_market_probability")
        best_price = market.get(f"{side}_best_price")
        margin_home = finite(score_profile.get("projected_margin_home"), 0.0)
        margin = margin_home if side == "home" else -margin_home
        h2h_delta = finite(h2h.get("h2h_delta"))
        if side == "away" and h2h_delta is not None:
            h2h_delta = -h2h_delta
        h2h_l5 = h2h.get("h2h_l5")
        h2h_l10 = h2h.get("h2h_l10")
        if side == "away":
            h2h_l5 = 1.0 - h2h_l5 if h2h_l5 is not None else None
            h2h_l10 = 1.0 - h2h_l10 if h2h_l10 is not None else None
        cards[side] = {
            "team": team, "opponent": opponent,
            "expected_value": expected_value_per_unit(ai, best_price),
            "probability_edge": (
                ai - market_probability
                if ai is not None and market_probability is not None else None
            ),
            "ai_probability": ai,
            "ai_probability_method": "confidence-adjusted model probability; not post-hoc calibrated",
            "base_projection": raw,
            "avg_margin": margin, "h2h_delta": h2h_delta,
            "h2h_l5": h2h_l5, "h2h_l10": h2h_l10,
            "pred_total": finite(score_profile.get("projected_total")),
            "pred_margin": margin,
            "proj_score": {
                "home": finite(score_profile.get("projected_home_runs")),
                "away": finite(score_profile.get("projected_away_runs")),
            },
            "market_implied_probability": market_probability,
            "market_price": best_price,
            "market_book_count": market.get("book_count", 0),
            "data_quality": {
                "evidence_factor": evidence,
                "signal_quality": finite(result.get("signal_quality")),
                "modern_games": int(finite(result.get(f"modern_games_{side}"), 0.0) or 0),
                "h2h_games": int(finite(h2h.get("h2h_games"), 0.0) or 0),
                "local_staleness_days": finite(local_staleness_days),
                "h2h_status": h2h.get("h2h_status", "NO_DATA"),
                "market_status": "SUPPORTED" if market.get("available") else "NO_MARKET",
                "market_price_filter": market.get("price_filter", "NO_MARKET"),
                "market_outlier_count": market.get("outlier_count", 0),
            },
            "source": "MLB Stats API current-season context + local pregame Statcast + market feed",
        }
    return cards


def build_batter_metric_card(
    *,
    batter_id: Any,
    batter_name: str,
    team: str,
    opponent_pitcher_id: Any,
    base_projection: Any,
    final_probability: Any,
    pa_df: pd.DataFrame,
    as_of_date: Any,
    local_staleness_days: Any = None,
    signal_quality: Any = 0.5,
    modern_player: Mapping[str, Any] | None = None,
    reference_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    form = batter_form_metrics(
        pa_df, batter_id, as_of_date,
        reference_context=reference_context,
        local_staleness_days=local_staleness_days,
    )
    h2h = batter_pitcher_h2h_metrics(
        pa_df, batter_id, opponent_pitcher_id, as_of_date
    )
    ai, evidence = confidence_adjusted_probability(
        final_probability,
        signal_quality=signal_quality,
        modern_games=(modern_player or {}).get("games_played", 0),
        local_staleness_days=local_staleness_days,
    )
    modern = modern_batter_context(
        modern_player, usage_pa_per_game=form.get("usage_pa_per_game")
    )
    return {
        "batter_id": finite(batter_id), "batter": batter_name, "team": team,
        "opponent_pitcher_id": finite(opponent_pitcher_id),
        "expected_value": None, "probability_edge": None,
        "ai_probability": ai,
        "ai_probability_method": "confidence-adjusted game-hit model; not post-hoc calibrated",
        "base_projection": finite(base_projection), "avg_margin": None,
        "h2h_delta": h2h.get("h2h_delta"), "h2h_l5": h2h.get("h2h_l5"),
        "h2h_l10": h2h.get("h2h_l10"),
        "last5": form.get("last5_hit_rate"), "last10": form.get("last10_hit_rate"),
        "last5_pa_hit_rate": form.get("last5_pa_hit_rate"),
        "last10_pa_hit_rate": form.get("last10_pa_hit_rate"),
        "usage_rtg": form.get("usage_rating"),
        "usage_pa_per_game": form.get("usage_pa_per_game"),
        "impact_rtg": form.get("impact_rating"),
        "min_ceiling": form.get("minimum_credible_hit_rate"),
        "sponge_coeff": None,
        "sponge_coeff_status": "UNDEFINED: no standard definition or validated source field",
        "eff_dropoff": form.get("efficiency_dropoff"),
        "l5_form_delta": form.get("last5_form_delta"),
        "l10_vs_prior10_form_delta": form.get("last10_vs_prior10_form_delta"),
        "modern_context": modern,
        "data_quality": {
            "evidence_factor": evidence,
            "local_staleness_days": finite(local_staleness_days),
            "last5_games": form.get("last5_games", 0),
            "last10_games": form.get("last10_games", 0),
            "h2h_pa": h2h.get("h2h_pa", 0),
            "h2h_games": h2h.get("h2h_games", 0),
            "form_status": form.get("data_status", "NO_DATA"),
            "h2h_status": h2h.get("h2h_status", "NO_DATA"),
            "modern_status": "SUPPORTED" if modern.get("available") else "NO_DATA",
        },
        "source": "local pregame Statcast; optional current-season MLB Stats API player context",
    }


def metric_definitions() -> dict[str, str]:
    return {
        "expected_value": "Expected profit per one unit staked at the best available price; null when no price exists.",
        "probability_edge": "AI probability minus median vig-adjusted market probability.",
        "ai_probability": "Confidence-adjusted model probability shrunk toward 50% for weak/stale evidence; not a fitted calibration curve.",
        "base_projection": "Raw trained-model probability before confidence adjustment.",
        "avg_margin": "Projected home score minus projected away score, signed from the card side.",
        "h2h_delta": "Shrunk directional H2H probability relative to neutral; requires actual completed matchup games.",
        "last5_last10": "Shrunk game-level hit rate (at least one hit in a game), with PA hit rates shown separately.",
        "usage_rtg": "Transparent proxy: recent PA per game divided by 4.2, capped at 100; not a lineup-order claim.",
        "impact_rtg": "0-100 relative contact-quality score from xwOBA/xBA/hard-hit against real local reference rates.",
        "min_ceiling": "Wilson lower confidence bound for the last-10 game hit rate; an uncertainty floor, not a guarantee.",
        "eff_dropoff": "Recent-10 minus prior-10 contact-quality delta using the first available real metric.",
        "modern_context": "Current-season MLB Stats API PA-rate context; game-hit conversion is explicitly labelled a proxy.",
        "sponge_coeff": "Not emitted as a number because the screenshot does not define a standard, validated statistic by this name.",
    }


def json_safe(value: Any) -> Any:
    """Recursively make cards safe for JSON output."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
