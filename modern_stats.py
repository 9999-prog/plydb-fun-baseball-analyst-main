"""Current-season team context from the public MLB Stats API.

The model's Statcast parquet is historical and can lag the prediction date.
This module provides a small, cached adapter for current-season team hitting
and pitching aggregates.  It intentionally returns an empty mapping on
network/API failure so callers can fall back to their historical features.
"""

from __future__ import annotations

from datetime import date
import math
from typing import Any

import requests


API_BASE = "https://statsapi.mlb.com/api/v1/teams/stats"
TEAMS_API = "https://statsapi.mlb.com/api/v1/teams"
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MODERN_WEIGHT = 0.70
MIN_MODERN_GAMES_FOR_FULL_WEIGHT = 10

_CACHE: dict[int, dict[str, dict[str, Any]]] = {}
_PLAYER_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_PLAYER_SEASON_CACHE: dict[int, dict[int, dict[str, Any]]] = {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_rate(numerator: Any, denominator: Any) -> float | None:
    numerator = _finite(numerator)
    denominator = _finite(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _season_from_value(value: Any | None) -> int:
    if value is None:
        return date.today().year
    return int(getattr(value, "year", value) if not isinstance(value, str) else value[:4])


def _fetch_team_abbreviations(season: int, timeout: int) -> dict[str, str]:
    """Map MLB team IDs to abbreviations used by schedules and local data."""
    try:
        response = requests.get(
            TEAMS_API,
            params={"sportId": 1, "season": season},
            timeout=timeout,
        )
        response.raise_for_status()
        teams = response.json().get("teams", [])
    except (requests.RequestException, ValueError, TypeError):
        return {}
    return {
        str(team.get("id")): str(team.get("abbreviation", "")).upper()
        for team in teams
        if team.get("id") is not None and team.get("abbreviation")
    }


def fetch_modern_team_stats(
    season: int | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Fetch and cache current-season team hitting/pitching aggregates.

    The returned mapping is keyed by uppercase team abbreviation.  A failure
    in either API request is non-fatal: partial data is retained, and a total
    failure returns ``{}`` so callers can use historical data.
    """
    season = _season_from_value(season)
    if season in _CACHE:
        return _CACHE[season]

    output: dict[str, dict[str, Any]] = {}
    team_abbreviations = _fetch_team_abbreviations(season, timeout)
    for group in ("hitting", "pitching"):
        try:
            response = requests.get(
                API_BASE,
                params={
                    "stats": "season",
                    "group": group,
                    "season": season,
                    "sportIds": 1,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            continue

        splits = []
        for stat_group in payload.get("stats", []):
            splits.extend(stat_group.get("splits", []))

        for split in splits:
            team = split.get("team", {})
            team_id = team.get("id")
            abbreviation = str(
                team.get("abbreviation")
                or team_abbreviations.get(str(team_id), "")
                or team.get("name", "")
            ).strip().upper()
            stat = split.get("stat", {})
            if not abbreviation:
                continue

            row = output.setdefault(
                abbreviation,
                {
                    "team": abbreviation,
                    "team_id": team.get("id"),
                    "season": season,
                },
            )
            games = _finite(stat.get("gamesPlayed"))
            runs = _finite(stat.get("runs"))
            row[f"{group}_games"] = int(games) if games is not None else 0

            if group == "hitting":
                row["runs_for"] = runs
                row["runs_per_game"] = _safe_rate(runs, games)
                row["hits"] = _finite(stat.get("hits"))
                row["at_bats"] = _finite(stat.get("atBats"))
                row["walks"] = _finite(stat.get("baseOnBalls"))
                row["strikeouts"] = _finite(stat.get("strikeOuts"))
                row["hit_rate"] = _safe_rate(row["hits"], row["at_bats"])
                row["walk_rate"] = _safe_rate(row["walks"], stat.get("plateAppearances"))
                row["strikeout_rate"] = _safe_rate(row["strikeouts"], stat.get("plateAppearances"))
                row["obp"] = _finite(stat.get("obp"))
                row["slg"] = _finite(stat.get("slg"))
                row["ops"] = _finite(stat.get("ops"))
            else:
                row["runs_allowed"] = runs
                row["runs_allowed_per_game"] = _safe_rate(runs, games)
                row["era"] = _finite(stat.get("era"))
                row["whip"] = _finite(stat.get("whip"))
                row["pitching_strikeouts"] = _finite(stat.get("strikeOuts"))
                row["pitching_walks"] = _finite(stat.get("baseOnBalls"))

    for row in output.values():
        hitting_games = int(row.get("hitting_games", 0))
        pitching_games = int(row.get("pitching_games", 0))
        if hitting_games and pitching_games:
            row["modern_games"] = min(hitting_games, pitching_games)
        else:
            row["modern_games"] = max(hitting_games, pitching_games)
        row["modern_available"] = bool(
            row.get("runs_per_game") is not None
            or row.get("runs_allowed_per_game") is not None
        )

    _CACHE[season] = output
    return output


def fetch_modern_player_stats(
    player_ids: Any,
    season: int | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[int, dict[str, Any]]:
    """Fetch current-season hitting stats for a small set of player IDs.

    The bulk Stats API endpoint is used once per season, then filtered locally.
    MLB reports real hits/PA and games played, but not the number of games in
    which a player had at least one hit. Callers must keep any conversion from
    PA rate to game-hit probability explicitly labelled as a proxy.
    """
    season = _season_from_value(season)
    ids = []
    for player_id in player_ids or []:
        try:
            value = int(float(player_id))
        except (TypeError, ValueError):
            continue
        if value not in ids:
            ids.append(value)

    if season not in _PLAYER_SEASON_CACHE:
        season_rows: dict[int, dict[str, Any]] = {}
        try:
            response = requests.get(
                "https://statsapi.mlb.com/api/v1/stats",
                params={
                    "stats": "season", "group": "hitting",
                    "season": season, "sportIds": 1, "limit": 1000,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            splits = []
            for stat_group in payload.get("stats", []):
                splits.extend(stat_group.get("splits", []))
            for split in splits:
                player_id = (split.get("player") or {}).get("id")
                stat = split.get("stat", {})
                try:
                    player_id = int(player_id)
                except (TypeError, ValueError):
                    continue
                games = _finite(stat.get("gamesPlayed"))
                pa = _finite(stat.get("plateAppearances"))
                hits = _finite(stat.get("hits"))
                season_rows[player_id] = {
                    "player_id": player_id, "season": season,
                    "available": bool(games is not None or pa is not None or hits is not None),
                    "games_played": int(games or 0), "plate_appearances": pa,
                    "at_bats": _finite(stat.get("atBats")), "hits": hits,
                    "home_runs": _finite(stat.get("homeRuns")),
                    "walks": _finite(stat.get("baseOnBalls")),
                    "strikeouts": _finite(stat.get("strikeOuts")),
                    "avg": _finite(stat.get("avg")),
                    "obp": _finite(stat.get("obp")),
                    "slg": _finite(stat.get("slg")),
                    "ops": _finite(stat.get("ops")),
                }
        except (requests.RequestException, ValueError, TypeError):
            season_rows = {}
        _PLAYER_SEASON_CACHE[season] = season_rows

    season_rows = _PLAYER_SEASON_CACHE[season]
    output: dict[int, dict[str, Any]] = {}
    for player_id in ids:
        cache_key = (season, player_id)
        if cache_key not in _PLAYER_CACHE:
            _PLAYER_CACHE[cache_key] = season_rows.get(
                player_id,
                {
                    "player_id": player_id, "season": season,
                    "available": False, "games_played": 0,
                    "plate_appearances": 0, "hits": 0,
                },
            )
        output[player_id] = _PLAYER_CACHE[cache_key]
    return output


def clear_cache() -> None:
    """Clear cached responses; useful for tests or a long-running refresh job."""
    _CACHE.clear()
    _PLAYER_CACHE.clear()
    _PLAYER_SEASON_CACHE.clear()


def blend_rate(
    historical_value: Any,
    historical_games: Any,
    modern_value: Any,
    modern_games: Any,
    *,
    modern_weight: float = DEFAULT_MODERN_WEIGHT,
    min_modern_games: int = MIN_MODERN_GAMES_FOR_FULL_WEIGHT,
) -> tuple[float | None, float]:
    """Blend one rate and return ``(value, effective_modern_weight)``.

    With at least ``min_modern_games`` of current-season data this is the
    requested 70% modern / 30% historical blend.  Smaller samples receive a
    proportional modern weight, which prevents a one-game box score from
    dominating the prediction.
    """
    historical = _finite(historical_value)
    modern = _finite(modern_value)
    historical_games = _finite(historical_games) or 0.0
    modern_games = _finite(modern_games) or 0.0

    if modern is not None and historical is not None and historical_games > 0:
        sample_confidence = min(1.0, max(0.0, modern_games / min_modern_games))
        effective_weight = max(0.0, min(1.0, modern_weight * sample_confidence))
        value = historical * (1.0 - effective_weight) + modern * effective_weight
        return value, effective_weight
    if modern is not None and modern_games > 0:
        return modern, 1.0
    if historical is not None and historical_games > 0:
        return historical, 0.0
    return None, 0.0
