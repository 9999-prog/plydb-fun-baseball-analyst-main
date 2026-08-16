"""UTC timestamp-safe snapshots for reproducible MLB pregame inference.

Timestamped records are eligible only when ``available_at < prediction_cutoff``.
Event time is never used as a substitute for publication/availability time.
All accepted timestamps must be explicit, timezone-aware values and are stored
as UTC.  The static adapter exists solely as a reversible compatibility path
for legacy joblib snapshots that have no availability provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class TemporalDataError(ValueError):
    """Raised when a temporal request cannot be made safely."""


_REQUIRED_AVAILABILITY_COLUMN = "available_at"
_OPTIONAL_TIME_COLUMNS = ("event_at", "ingested_at")


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    """Parse one explicit timezone-aware timestamp and normalize it to UTC."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TemporalDataError(f"Invalid {label}: {value!r}") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise TemporalDataError(f"{label} must be timezone-aware: {value!r}")
    try:
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise TemporalDataError(f"Invalid {label}: {value!r}") from exc


def _validated_utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse a timestamp column without silently accepting bad records."""
    if column not in frame.columns:
        raise TemporalDataError(f"Missing required timestamp column: {column}")
    values = []
    for index, value in frame[column].items():
        try:
            values.append(_utc_timestamp(value, label=f"{column} at row {index}"))
        except TemporalDataError as exc:
            raise TemporalDataError(str(exc)) from exc
    return pd.Series(values, index=frame.index, dtype="datetime64[ns, UTC]")


def _validate_optional_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    for column in _OPTIONAL_TIME_COLUMNS:
        if column in result.columns:
            result[column] = _validated_utc_series(result, column)
    return result


def as_of_filter(
    frame: pd.DataFrame,
    prediction_cutoff: object,
    *,
    availability_column: str = _REQUIRED_AVAILABILITY_COLUMN,
) -> pd.DataFrame:
    """Return a new frame containing only data published before the cutoff.

    The comparison is strict: records exactly at the cutoff are excluded.
    Neither calendar dates nor file metadata are considered valid availability
    evidence. Missing, malformed, timezone-naive, and ambiguous timestamps
    raise ``TemporalDataError`` rather than being guessed or silently dropped.
    """
    cutoff = _utc_timestamp(prediction_cutoff, label="prediction cutoff")
    result = _validate_optional_time_columns(frame)
    result[availability_column] = _validated_utc_series(result, availability_column)
    result = result.loc[result[availability_column].lt(cutoff)].copy()
    result["prediction_cutoff"] = cutoff
    return result


def _tie_break_hash(frame: pd.DataFrame) -> pd.Series:
    """Produce an input-order-independent final tie breaker for duplicate rows."""
    columns = sorted(column for column in frame.columns if column != "prediction_cutoff")

    def row_hash(row: pd.Series) -> str:
        payload = "\x1f".join(f"{column}={row[column]!r}" for column in columns)
        return sha256(payload.encode("utf-8")).hexdigest()

    return frame.apply(row_hash, axis=1)


def latest_entity_snapshot(
    frame: pd.DataFrame,
    *,
    entity_columns: Iterable[str],
    prediction_cutoff: object,
    value_columns: Iterable[str] | None = None,
    availability_column: str = _REQUIRED_AVAILABILITY_COLUMN,
    entity_filter: Mapping[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """Select the latest deterministically ordered eligible record per entity.

    Revision records are ordered first by when they became available, then by
    event time, ingestion time, revision/version, source, and a stable row
    hash. A late-published correction therefore cannot appear in a historical
    prediction before its own ``available_at`` timestamp.
    """
    entity_columns = list(entity_columns)
    missing = [column for column in entity_columns if column not in frame.columns]
    if missing:
        raise TemporalDataError(f"Missing entity columns: {', '.join(missing)}")

    history = as_of_filter(
        frame,
        prediction_cutoff,
        availability_column=availability_column,
    )
    if entity_filter:
        for column, values in entity_filter.items():
            if column not in history.columns:
                raise TemporalDataError(f"Unknown entity filter column: {column}")
            history = history.loc[history[column].isin(list(values))].copy()

    selected = list(frame.columns) if value_columns is None else list(dict.fromkeys([
        *entity_columns,
        availability_column,
        *[column for column in _OPTIONAL_TIME_COLUMNS if column in frame.columns],
        *[column for column in ("source", "revision", "version") if column in frame.columns],
        *[column for column in value_columns if column in frame.columns],
    ]))
    if history.empty:
        return pd.DataFrame(columns=[*selected, "prediction_cutoff"])

    history = history[selected + ["prediction_cutoff"]].copy()
    history["_tie_break_hash"] = _tie_break_hash(history)
    sort_columns = [*entity_columns, availability_column]
    for column in ("event_at", "ingested_at", "revision", "version", "source", "_tie_break_hash"):
        if column in history.columns:
            sort_columns.append(column)
    snapshot = (
        history.sort_values(sort_columns, kind="mergesort", na_position="first")
        .drop_duplicates(entity_columns, keep="last")
        .drop(columns="_tie_break_hash")
        .reset_index(drop=True)
    )
    return snapshot


@dataclass(frozen=True)
class TimestampedSnapshotAdapter:
    """Adapter for provenance-complete historical snapshot records."""

    records: pd.DataFrame
    entity_columns: tuple[str, ...]
    availability_column: str = _REQUIRED_AVAILABILITY_COLUMN

    def snapshot(
        self,
        prediction_cutoff: object,
        *,
        entity_filter: Mapping[str, Sequence[Any]] | None = None,
        value_columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return latest_entity_snapshot(
            self.records,
            entity_columns=self.entity_columns,
            prediction_cutoff=prediction_cutoff,
            availability_column=self.availability_column,
            entity_filter=entity_filter,
            value_columns=value_columns,
        )


@dataclass(frozen=True)
class StaticSnapshotAdapter:
    """Reversible legacy adapter; unsuitable for historical backtests.

    Legacy snapshots do not contain availability metadata, so this adapter
    deliberately does not pretend to perform an as-of selection. It validates
    the explicit UTC cutoff, returns a copy, and marks the result as static.
    """

    snapshots: Mapping[str, pd.DataFrame]

    def snapshot(self, name: str, prediction_cutoff: object) -> pd.DataFrame:
        cutoff = _utc_timestamp(prediction_cutoff, label="prediction cutoff")
        if name not in self.snapshots:
            return pd.DataFrame()
        snapshot = self.snapshots[name]
        if not isinstance(snapshot, pd.DataFrame):
            raise TemporalDataError(f"Static snapshot {name!r} is not a DataFrame")
        result = snapshot.copy(deep=True)
        result["prediction_cutoff"] = cutoff
        result["snapshot_provenance"] = "legacy_static_rollback"
        return result
