"""Central configuration loader for MLB Baseball Analyst."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_PATH: Path | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache configuration from YAML file."""
    global _CONFIG_CACHE, _CONFIG_PATH
    
    if _CONFIG_CACHE is not None and _CONFIG_PATH == Path(config_path) if config_path else True:
        return _CONFIG_CACHE
    
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    _CONFIG_PATH = config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    
    # Allow environment variable overrides for sensitive values
    _apply_env_overrides(_CONFIG_CACHE)
    
    return _CONFIG_CACHE


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Apply environment variable overrides to config."""
    # ODDS API key
    odds_key = os.getenv("ODDS_API_KEY", "").strip()
    if odds_key:
        config.setdefault("odds", {})["api_key"] = odds_key
    
    # Log level
    log_level = os.getenv("LOG_LEVEL", "").strip().upper()
    if log_level:
        config.setdefault("logging", {})["level"] = log_level
    
    # Disable colors
    if os.getenv("NO_COLOR") or os.getenv("PREDICTOR_COLOR", "").strip() in {"0", "false", "no"}:
        config.setdefault("logging", {})["console_colors"] = False


def get_config() -> dict[str, Any]:
    """Get cached config, loading if necessary."""
    if _CONFIG_CACHE is None:
        return load_config()
    return _CONFIG_CACHE


def reset_config() -> None:
    """Reset cached config (useful for tests)."""
    global _CONFIG_CACHE, _CONFIG_PATH
    _CONFIG_CACHE = None
    _CONFIG_PATH = None


# Convenience accessors
def get_odds_config() -> dict[str, Any]:
    return get_config().get("odds", {})


def get_win_model_weights() -> dict[str, Any]:
    return get_config().get("win_model_weights", {})


def get_batter_model_weights() -> dict[str, Any]:
    return get_config().get("batter_model_weights", {})


def get_shrinkage_params() -> dict[str, Any]:
    return get_config().get("shrinkage", {})


def get_validation_params() -> dict[str, Any]:
    return get_config().get("validation", {})


def get_backtest_params() -> dict[str, Any]:
    return get_config().get("backtest", {})


def get_data_freshness_params() -> dict[str, Any]:
    return get_config().get("data_freshness", {})


def get_feature_params() -> dict[str, Any]:
    return get_config().get("features", {})


def get_logging_params() -> dict[str, Any]:
    return get_config().get("logging", {})


def get_paths() -> dict[str, Any]:
    return get_config().get("paths", {})


def get_timezone() -> str:
    return get_config().get("timezone", "America/New_York")