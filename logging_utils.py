"""Structured logging for MLB Baseball Analyst."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config_loader import get_logging_params


class StructuredFormatter(logging.Formatter):
    """JSON formatter with consistent fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info"
            }:
                log_entry[key] = value
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for console."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        name = record.name
        message = record.getMessage()
        return f"{timestamp} [{level:8s}] {name}: {message}"


def setup_logging(name: str = "mlb_analyst") -> logging.Logger:
    """Configure structured logging for the application."""
    params = get_logging_params()
    log_level = getattr(logging, params.get("level", "INFO").upper(), logging.INFO)
    log_format = params.get("format", "json")
    log_file = params.get("file", "logs/predictor.log")
    console_colors = params.get("console_colors", True)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if log_format == "json":
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(TextFormatter())
    
    logger.addHandler(console_handler)
    
    # File handler (always JSON for parsing)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "mlb_analyst") -> logging.Logger:
    """Get logger instance, setting up if needed."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


class PredictionLogger:
    """Specialized logger for prediction events with structured output."""
    
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or get_logger("mlb_analyst.predictions")
    
    def log_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        prediction_type: str,  # "win", "total", "batter_hit"
        model_prob: float,
        market_prob: float | None,
        edge: float | None,
        ev: float | None,
        decision: str,  # "BET", "NO_BET"
        kelly_fraction: float | None = None,
        signal_quality: float | None = None,
        staleness_days: int | None = None,
        odds_timestamp: str | None = None,
        **extra
    ):
        self.logger.info(
            "prediction",
            extra={
                "event_type": "prediction",
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "prediction_type": prediction_type,
                "model_probability": model_prob,
                "market_probability": market_prob,
                "edge_pp": round(edge * 100, 1) if edge is not None else None,
                "expected_value": ev,
                "decision": decision,
                "kelly_fraction": kelly_fraction,
                "signal_quality": signal_quality,
                "data_staleness_days": staleness_days,
                "odds_timestamp": odds_timestamp,
                **extra
            }
        )
    
    def log_model_training(
        self,
        model_name: str,
        train_seasons: list[int],
        test_season: int,
        metrics: dict[str, float],
        n_train: int,
        n_test: int,
        **extra
    ):
        self.logger.info(
            "model_training",
            extra={
                "event_type": "model_training",
                "model_name": model_name,
                "train_seasons": train_seasons,
                "test_season": test_season,
                "metrics": metrics,
                "n_train": n_train,
                "n_test": n_test,
                **extra
            }
        )
    
    def log_data_ingestion(
        self,
        source: str,
        records: int,
        date_range: tuple[str, str] | None = None,
        staleness_days: int | None = None,
        **extra
    ):
        self.logger.info(
            "data_ingestion",
            extra={
                "event_type": "data_ingestion",
                "source": source,
                "records": records,
                "date_range_start": date_range[0] if date_range else None,
                "date_range_end": date_range[1] if date_range else None,
                "staleness_days": staleness_days,
                **extra
            }
        )
    
    def log_api_call(
        self,
        api: str,
        endpoint: str,
        status: int,
        latency_ms: float,
        **extra
    ):
        self.logger.info(
            "api_call",
            extra={
                "event_type": "api_call",
                "api": api,
                "endpoint": endpoint,
                "status_code": status,
                "latency_ms": latency_ms,
                **extra
            }
        )


# Timer context manager for logging operation duration
class LogTimer:
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        level = logging.ERROR if exc_type else logging.INFO
        self.logger.log(
            level,
            f"{self.operation} completed",
            extra={
                "event_type": "operation_timing",
                "operation": self.operation,
                "duration_ms": round(duration_ms, 1),
                "success": exc_type is None,
                **self.context
            }
        )