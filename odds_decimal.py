"""Decimal odds utilities - replaces American odds throughout the codebase.

Decimal odds are mathematically cleaner:
- Implied probability = 1 / decimal_odds
- EV = model_prob * decimal_odds - 1
- No piecewise conversion needed
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def decimal_to_probability(decimal_odds: Any) -> float | None:
    """Convert decimal odds to implied probability.
    
    Args:
        decimal_odds: Decimal odds (e.g., 1.91 for -110 American)
        
    Returns:
        Implied probability (0-1) or None if invalid
    """
    try:
        odds = float(decimal_odds)
        if odds <= 1.0 or not math.isfinite(odds):
            return None
        return 1.0 / odds
    except (TypeError, ValueError):
        return None


def probability_to_decimal(probability: Any) -> float | None:
    """Convert probability to fair decimal odds.
    
    Args:
        probability: Probability (0-1)
        
    Returns:
        Fair decimal odds or None if invalid
    """
    try:
        p = float(probability)
        if p <= 0.0 or p >= 1.0 or not math.isfinite(p):
            return None
        return 1.0 / p
    except (TypeError, ValueError):
        return None


def expected_value_decimal(model_prob: Any, decimal_odds: Any) -> float | None:
    """Calculate expected value per unit staked at decimal odds.
    
    EV = model_prob * decimal_odds - 1
    
    Args:
        model_prob: Model's estimated probability (0-1)
        decimal_odds: Decimal odds offered
        
    Returns:
        Expected value per unit (positive = edge) or None if invalid
    """
    try:
        p = float(model_prob)
        odds = float(decimal_odds)
        if not (0.0 < p < 1.0) or odds <= 1.0:
            return None
        if not (math.isfinite(p) and math.isfinite(odds)):
            return None
        return p * odds - 1.0
    except (TypeError, ValueError):
        return None


def kelly_fraction(model_prob: Any, decimal_odds: Any, fraction: float = 1.0) -> float | None:
    """Calculate Kelly criterion bet fraction.
    
    Kelly = (p * b - q) / b where b = decimal_odds - 1, q = 1 - p
    Simplified: (p * odds - 1) / (odds - 1)
    
    Args:
        model_prob: Model's estimated probability (0-1)
        decimal_odds: Decimal odds offered
        fraction: Fraction of Kelly to use (e.g., 0.25 for 1/4 Kelly)
        
    Returns:
        Optimal bet fraction of bankroll or None if invalid
    """
    try:
        p = float(model_prob)
        odds = float(decimal_odds)
        if not (0.0 < p < 1.0) or odds <= 1.0:
            return None
        if not (math.isfinite(p) and math.isfinite(odds)):
            return None
        
        # Full Kelly
        kelly = (p * odds - 1.0) / (odds - 1.0)
        
        # Apply fraction
        kelly = kelly * fraction
        
        # Cap at reasonable bounds
        return max(0.0, min(kelly, 1.0))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def no_vig_probabilities(home_decimal: Any, away_decimal: Any) -> tuple[float | None, float | None]:
    """Calculate no-vig (fair) probabilities from decimal odds.
    
    Args:
        home_decimal: Home team decimal odds
        away_decimal: Away team decimal odds
        
    Returns:
        Tuple of (home_no_vig_prob, away_no_vig_prob) or (None, None)
    """
    home_raw = decimal_to_probability(home_decimal)
    away_raw = decimal_to_probability(away_decimal)
    
    if home_raw is None or away_raw is None:
        return None, None
    
    total = home_raw + away_raw
    if total <= 0:
        return None, None
    
    return home_raw / total, away_raw / total


def no_vig_from_multiple(odds_list: list[dict[str, float]]) -> tuple[float | None, float | None]:
    """Calculate consensus no-vig probabilities from multiple bookmakers.
    
    Args:
        odds_list: List of dicts with 'home_decimal' and 'away_decimal' keys
        
    Returns:
        Tuple of (median_home_no_vig, median_away_no_vig)
    """
    home_probs = []
    away_probs = []
    
    for odds in odds_list:
        h, a = no_vig_probabilities(odds.get("home_decimal"), odds.get("away_decimal"))
        if h is not None and a is not None:
            home_probs.append(h)
            away_probs.append(a)
    
    if not home_probs:
        return None, None
    
    return float(np.median(home_probs)), float(np.median(away_probs))


def best_decimal_odds(odds_list: list[dict[str, float]], side: str = "home") -> float | None:
    """Get best available decimal odds for a side.
    
    Args:
        odds_list: List of dicts with 'home_decimal' and 'away_decimal' keys
        side: "home" or "away"
        
    Returns:
        Best (highest) decimal odds for the side
    """
    key = "home_decimal" if side == "home" else "away_decimal"
    valid_odds = [odds[key] for odds in odds_list if odds.get(key) and odds[key] > 1.0]
    
    if not valid_odds:
        return None
    
    return max(valid_odds)


def american_to_decimal(american_odds: Any) -> float | None:
    """Convert American odds to decimal (for API compatibility)."""
    try:
        odds = float(american_odds)
        if odds > 0:
            return 1.0 + odds / 100.0
        elif odds < 0:
            return 1.0 + 100.0 / abs(odds)
        return None
    except (TypeError, ValueError):
        return None


def decimal_to_american(decimal_odds: Any) -> int | None:
    """Convert decimal odds to American (for display compatibility)."""
    try:
        odds = float(decimal_odds)
        if odds <= 1.0:
            return None
        if odds >= 2.0:
            return int(round((odds - 1.0) * 100))
        else:
            return int(round(-100.0 / (odds - 1.0)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# For backward compatibility during transition
def american_to_probability(odds: Any) -> float | None:
    """Convert American odds to implied probability (DEPRECATED)."""
    decimal = american_to_decimal(odds)
    return decimal_to_probability(decimal)


def fair_american_odds(probability: Any) -> int | None:
    """Convert probability to fair American odds (DEPRECATED)."""
    decimal = probability_to_decimal(probability)
    return decimal_to_american(decimal)


def expected_value_per_unit(probability: Any, american_odds: Any) -> float | None:
    """Expected value with American odds (DEPRECATED)."""
    decimal = american_to_decimal(american_odds)
    return expected_value_decimal(probability, decimal)


# Validation helpers
def validate_decimal_odds(odds: Any) -> bool:
    """Check if value is valid decimal odds."""
    try:
        val = float(odds)
        return val > 1.0 and math.isfinite(val)
    except (TypeError, ValueError):
        return False


def clamp_probability(p: Any, eps: float = 1e-6) -> float:
    """Clamp probability to valid range."""
    try:
        val = float(p)
        if not math.isfinite(val):
            return 0.5
        return max(eps, min(1.0 - eps, val))
    except (TypeError, ValueError):
        return 0.5


def logit(p: Any, eps: float = 1e-4) -> float:
    """Logit transform with clamping."""
    p = clamp_probability(p, eps)
    return math.log(p / (1.0 - p))


def inv_logit(x: Any) -> float:
    """Inverse logit (sigmoid)."""
    try:
        val = float(x)
        if not math.isfinite(val):
            return 0.5
        return 1.0 / (1.0 + math.exp(-val))
    except (TypeError, ValueError, OverflowError):
        return 0.5