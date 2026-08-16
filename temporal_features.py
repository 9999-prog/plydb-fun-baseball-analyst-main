"""Timestamp-safe feature helpers for MLB pregame inference.

All helpers fail closed: data with an unknown or invalid event timestamp is not
used to construct a historical prediction feature.  Callers must provide the
prediction's as-of timestamp rather than relying on a mutable global date.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


class TemporalDataError(ValueError):
    """Raised when an as-of feature request cannot be made safely."""


def _timestamp(value: object, *, label: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise TemporalDataError(f"Invalid {label}: {value!r}")
    return timestamp


def as_of_filter(
    frame: pd.DataFrame,
    as_of_timestamp: object,
    *,
    timestamp_column: str = "game_date",
) -> pd.DataFrame:
    """Return a copy containing records strictly available before ``as_of``.

    Dates without an event time are normalized to midnight UTC. Therefore a
    pregame request at midnight excludes all games on that calendar date,
    avoiding accidental same-day/postgame leakage.  A later implementation may
    pass true game-start timestamps for more granular historical replay.
    """
    if timestamp_column not in frame.columns:
        raise TemporalDataError(f"Missing required timestamp column: {timestamp_column}")

    cutoff = _timestamp(as_of_timestamp, label="as-of timestamp")
    event_time = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    valid_rows = event_time.notna() & event_time.lt(cutoff)
    result = frame.loc[valid_rows].copy()
    result[timestamp_column] = event_time.loc[valid_rows]
    return result


def latest_entity_snapshot(
    frame: pd.DataFrame,
    *,
    entity_columns: Iterable[str],
    as_of_timestamp: object,
    value_columns: Iterable[str] | None = None,
    timestamp_column: str = "game_date",
    game_id_column: str = "game_pk",
) -> pd.DataFrame:
    """Return each entity's latest row known strictly before ``as_of``.

    Stable sorting makes same-date rows deterministic.  A game identifier is
    required as a secondary ordering key because parquet row order is not a
    reproducibility contract.
    """
    entity_columns = list(entity_columns)
    required = [timestamp_column, game_id_column, *entity_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise TemporalDataError(f"Missing required snapshot columns: {', '.join(missing)}")

    history = as_of_filter(
        frame,
        as_of_timestamp,
        timestamp_column=timestamp_column,
    )
    if history.empty:
        columns = [*entity_columns, game_id_column, timestamp_column]
        if value_columns:
            columns.extend(column for column in value_columns if column not in columns)
        return pd.DataFrame(columns=columns)

    selected_columns = [*entity_columns, game_id_column, timestamp_column]
    if value_columns is not None:
        selected_columns.extend(column for column in value_columns if column in history.columns)
    selected_columns = list(dict.fromkeys(selected_columns))
    history = history[selected_columns]
    return (
        history.sort_values(
            [*entity_columns, timestamp_column, game_id_column],
            kind="mergesort",
        )
        .drop_duplicates(entity_columns, keep="last")
        .reset_index(drop=True)
    )
