"""Side-effect-free safety and presentation helpers for the MLB predictor."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Iterable


_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|token|secret|password|access[_-]?token)=)[^&#\s]+"
)
_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|access[_-]?token)\b\s*[:=]\s*[^\s,;]+"
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_error(error: BaseException, *, secrets: Iterable[Any] = ()) -> str:
    """Format an exception without exposing credentials or query strings."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status in {401, 403}:
        return f"HTTP {int(status)} credentials rejected"

    message = str(error).strip() or type(error).__name__
    values = list(secrets)
    configured = os.getenv("ODDS_API_KEY", "")
    if configured:
        values.append(configured)
    for secret in values:
        secret_text = str(secret).strip()
        if len(secret_text) >= 4:
            message = message.replace(secret_text, "[REDACTED]")
    message = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", message)
    message = _KEY_VALUE_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", message)
    return message


def _probability(value: Any) -> float | None:
    value = _finite(value)
    if value is None or value < 0.0 or value > 1.0:
        return None
    return value


def pick_annotation(
    probability: Any,
    edge: Any = None,
    signal_quality: Any = 1.0,
    *,
    market_available: bool | None = None,
    weak_threshold: float = 0.52,
    strong_threshold: float = 0.63,
) -> str:
    """Return a label that never claims value when no market exists."""
    probability = _probability(probability)
    if probability is None:
        return "PASS - NO PROBABILITY"

    quality = _finite(signal_quality)
    quality = 0.0 if quality is None else max(0.0, min(1.0, quality))
    if quality < 0.35:
        return "PASS - WEAK DATA"

    edge = _finite(edge)
    if market_available is None:
        market_available = edge is not None
    if market_available:
        if edge is not None and edge >= 0.04:
            return "VALUE EDGE"
        if edge is not None and edge >= 0.015:
            return "MARKET EDGE"
        if edge is not None and edge <= -0.02:
            return "PASS - BAD PRICE"

    if probability >= strong_threshold:
        return "STRONG MODEL LEAN"
    if probability >= weak_threshold:
        return "MODEL LEAN"
    return "PASS - THIN LEAN"


def clean_sentence(text: Any) -> str:
    """Normalize generated prose so it cannot end in doubled punctuation."""
    value = re.sub(r"\.{2,}", ".", str(text))
    return re.sub(r"\s+", " ", value).strip()
