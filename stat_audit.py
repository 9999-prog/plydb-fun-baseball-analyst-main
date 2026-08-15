"""Audit model-ready baseball data and report statistical risk.

This is intentionally separate from training/prediction so it can be run
before either one.  It checks freshness, missingness, duplicate keys, target
balance, constant features, likely post-game leakage, calibration, and the
model's in-sample diagnostics.  In-sample metrics are labeled as such: they
are not a substitute for a time-based holdout.

Examples:
    python stat_audit.py
    python stat_audit.py --dataset data/pybaseball/statcast/statcast_multiseason_batter_game_level.parquet
    python stat_audit.py --json-out audit.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_DATASET = os.path.join(
    "data",
    "pybaseball",
    "statcast",
    "statcast_multiseason_batter_game_level.parquet",
)
DEFAULT_MODEL = "hit_model.joblib"

# These columns are outcomes or post-event observations.  They must not be in
# a pregame feature list, even if they happen to correlate strongly with wins.
POSTGAME_COLUMNS = {
    "events",
    "description",
    "is_hit",
    "is_walk",
    "is_strikeout",
    "is_hbp",
    "hit_in_game",
    "home_win",
    "bat_score",
    "post_bat_score",
    "post_home_score",
    "post_away_score",
    "runs_on_pa",
    "rbi",
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def likely_leakage(features: list[str]) -> list[str]:
    """Return feature names that look like post-game information."""
    found = []
    for feature in features:
        name = feature.lower()
        if feature in POSTGAME_COLUMNS or any(
            token in name
            for token in (
                "post_",
                "final_",
                "outcome",
                "target",
                "score_after",
            )
        ):
            found.append(feature)
    return found


def calibration_table(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Return count, mean prediction, event rate, and calibration gap."""
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(actual, errors="coerce"),
            "probability": pd.to_numeric(probabilities, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=["bin", "count", "mean_predicted", "event_rate", "gap"]
        )

    frame["probability"] = frame["probability"].clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    table = (
        frame.groupby("bin", observed=False)
        .agg(
            count=("actual", "size"),
            mean_predicted=("probability", "mean"),
            event_rate=("actual", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["event_rate"] - table["mean_predicted"]
    # Pandas interval objects are not JSON-native; keep the report portable.
    table["bin"] = table["bin"].astype(str)
    return table


def classifier_metrics(actual, probabilities) -> dict[str, float | None]:
    """Compute metrics without making sklearn a hard dependency."""
    y = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(probabilities), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return {"rows": 0, "roc_auc": None, "brier": None, "log_loss": None}

    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auc = None
    if len(np.unique(y)) == 2:
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        positive = y == 1
        negative = y == 0
        auc = float(
            (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
            / (positive.sum() * negative.sum())
        )
    return {
        "rows": int(len(y)),
        "roc_auc": auc,
        "brier": brier,
        "log_loss": log_loss,
    }


def audit_dataframe(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Create a JSON-friendly structural/statistical audit."""
    date_columns = [column for column in ("game_date", "date") if column in df.columns]
    latest = earliest = None
    freshness_days = None
    if date_columns:
        dates = pd.to_datetime(df[date_columns[0]], errors="coerce").dropna()
        if not dates.empty:
            earliest = dates.min()
            latest = dates.max()
            now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
            if getattr(latest, "tzinfo", None) is not None:
                latest = latest.tz_localize(None)
            freshness_days = int(max(0, (now - latest).days))

    freshness_status = None
    if freshness_days is not None:
        freshness_status = (
            "fresh" if freshness_days <= 7
            else "aging" if freshness_days <= 30
            else "stale"
        )
    audit: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": _json_value(earliest),
        "date_max": _json_value(latest),
        "freshness_days": freshness_days,
        "freshness_status": freshness_status,
        "seasons": sorted(
            int(value)
            for value in pd.to_numeric(df["season"], errors="coerce").dropna().unique()
        )
        if "season" in df.columns
        else [],
        "missing_top": {
            str(column): float(value)
            for column, value in df.isna().mean().sort_values(ascending=False).head(15).items()
            if value > 0
        },
        "constant_columns": [
            str(column) for column in df.columns if df[column].nunique(dropna=True) <= 1
        ],
        "duplicate_rows": int(df.duplicated().sum()),
    }

    key_candidates = [
        ["game_pk", "batter"],
        ["game_pk", "home_team", "away_team"],
    ]
    audit["duplicate_keys"] = {}
    for keys in key_candidates:
        if all(key in df.columns for key in keys):
            audit["duplicate_keys"]["+".join(keys)] = int(
                df.duplicated(keys).sum()
            )

    if target and target in df.columns:
        target_values = pd.to_numeric(df[target], errors="coerce").dropna()
        audit["target"] = {
            "name": target,
            "rows": int(len(target_values)),
            "mean": float(target_values.mean()) if not target_values.empty else None,
            "unique": sorted(_json_value(value) for value in target_values.unique()),
        }

    if feature_columns:
        missing = sorted(set(feature_columns).difference(df.columns))
        audit["features"] = {
            "requested": len(feature_columns),
            "missing": missing,
            "leakage_candidates": likely_leakage(feature_columns),
            "missing_rate": {
                column: float(df[column].isna().mean())
                for column in feature_columns
                if column in df.columns
            },
        }
    return audit


def model_audit(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    """Run explicitly labeled in-sample diagnostics for a saved artifact."""
    features = list(artifact.get("features", []))
    report: dict[str, Any] = {
        "feature_count": len(features),
        "leakage_candidates": likely_leakage(features),
    }
    missing = sorted(set(features + [target]).difference(df.columns))
    report["missing_columns"] = missing
    if missing or "model" not in artifact:
        return report

    usable = df.dropna(subset=features + [target]).copy()
    if usable.empty:
        report["metrics"] = classifier_metrics([], [])
        return report

    model = artifact["model"]
    probabilities = model.predict_proba(usable[features])[:, 1]
    report["metrics"] = classifier_metrics(usable[target], probabilities)
    baseline_probability = float(pd.to_numeric(usable[target], errors="coerce").mean())
    report["baseline_metrics"] = classifier_metrics(
        usable[target], np.full(len(usable), baseline_probability)
    )
    report["calibration"] = calibration_table(
        usable[target], probabilities
    ).to_dict(orient="records")
    if hasattr(model, "feature_importances_"):
        report["feature_importance"] = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(features, model.feature_importances_)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )
    report["warning"] = (
        "These metrics are in-sample because the saved artifact was trained on "
        "the same table. Use a season-held-out rebuild for honest performance."
    )
    return report



def temporal_holdout_audit(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    """Fit a fresh copy of the saved estimator on prior seasons only.

    This is intentionally separate from ``model_audit``.  The saved artifact
    is normally trained on all available rows, so its in-sample score can be
    useful for smoke-testing but cannot estimate future performance.  This
    check trains on every season before the latest season and evaluates only
    that latest season, preserving the time direction of a real forecast.
    """
    features = list(artifact.get("features", []))
    result: dict[str, Any] = {
        "method": "train_on_prior_seasons_test_on_latest",
        "feature_count": len(features),
    }
    if "season" in df.columns:
        seasons = sorted(
            int(value)
            for value in pd.to_numeric(df["season"], errors="coerce").dropna().unique()
        )
    elif "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"], errors="coerce")
        seasons = sorted(int(value) for value in dates.dt.year.dropna().unique())
    else:
        result["warning"] = "No season or game_date column is available for a time split."
        return result

    if len(seasons) < 2:
        result["seasons"] = seasons
        result["warning"] = "At least two seasons are required for a temporal holdout."
        return result

    missing = sorted(set(features + [target]).difference(df.columns))
    if missing or "model" not in artifact:
        result["missing_columns"] = missing
        return result

    work = df.copy()
    work["_audit_season"] = (
        pd.to_numeric(work["season"], errors="coerce")
        if "season" in work.columns
        else pd.to_datetime(work["game_date"], errors="coerce").dt.year
    )
    work = work.dropna(subset=features + [target, "_audit_season"])
    latest_season = seasons[-1]
    train = work[work["_audit_season"] < latest_season]
    test = work[work["_audit_season"] == latest_season]
    result.update(
        {
            "train_seasons": [int(value) for value in seasons[:-1]],
            "test_season": int(latest_season),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )
    if train.empty or test.empty:
        result["warning"] = "The temporal split has no train or test rows."
        return result
    if pd.to_numeric(train[target], errors="coerce").nunique() < 2:
        result["warning"] = "The temporal training target has only one class."
        return result

    try:
        from sklearn.base import clone

        validation_model = clone(artifact["model"])
    except Exception:
        # A deep copy is a safe fallback for estimators that are not sklearn
        # clone-compatible, while still keeping the saved artifact untouched.
        import copy

        validation_model = copy.deepcopy(artifact["model"])

    try:
        validation_model.fit(train[features], train[target])
        probabilities = validation_model.predict_proba(test[features])[:, 1]
    except Exception as exc:  # noqa: BLE001
        result["warning"] = f"Temporal estimator fit/predict failed: {exc}"
        return result

    result["metrics"] = classifier_metrics(test[target], probabilities)
    train_rate = float(pd.to_numeric(train[target], errors="coerce").mean())
    result["baseline_metrics"] = classifier_metrics(
        test[target], np.full(len(test), train_rate)
    )
    result["calibration"] = calibration_table(
        test[target], probabilities
    ).to_dict(orient="records")
    result["warning"] = (
        "This is a latest-season holdout, not a live betting guarantee. It still "
        "assumes the feature table itself was built without post-game leakage."
    )
    return result


def print_report(report: dict[str, Any]) -> None:
    print("STATISTICAL AUDIT")
    print("=" * 80)
    print(f"Rows: {report.get('rows', 0):,} | Columns: {report.get('columns', 0):,}")
    print(
        f"Coverage: {report.get('date_min', 'N/A')} -> {report.get('date_max', 'N/A')} "
        f"({report.get('freshness_days', 'N/A')} days old; "
        f"{report.get('freshness_status', 'unknown')})"
    )
    if report.get("freshness_status") == "stale":
        print("WARNING: local data is stale; use live current-season context and lower confidence.")
    print(f"Seasons: {report.get('seasons') or 'N/A'}")
    print(f"Duplicate rows: {report.get('duplicate_rows', 0):,}")

    target = report.get("target")
    if target:
        print(
            f"Target {target['name']}: {target['rows']:,} rows, "
            f"mean={target['mean'] if target['mean'] is not None else 'N/A'}"
        )

    features = report.get("features", {})
    if features:
        print(f"Features requested: {features.get('requested', 0)}")
        print(f"Missing features: {features.get('missing') or 'none'}")
        print(
            "Leakage candidates: "
            f"{features.get('leakage_candidates') or 'none'}"
        )

    print(f"Constant columns: {report.get('constant_columns') or 'none'}")
    print("Missingness (top):")
    for column, rate in report.get("missing_top", {}).items():
        print(f"  {column}: {rate:.1%}")

    model = report.get("model")
    if model:
        print("\nMODEL CHECK (IN-SAMPLE ONLY)")
        print(f"Metrics: {model.get('metrics')}")
        print(f"Baseline: {model.get('baseline_metrics')}")
        print(model.get("warning", ""))
        temporal = report.get("temporal_validation")
        if temporal:
            print("\nTIME-HOLDOUT CHECK")
            print(
                f"Train seasons: {temporal.get('train_seasons')} -> "
                f"test season: {temporal.get('test_season')}"
            )
            print(
                f"Rows: {temporal.get('train_rows', 0):,} train / "
                f"{temporal.get('test_rows', 0):,} test"
            )
            print(f"Metrics: {temporal.get('metrics')}")
            print(f"Baseline: {temporal.get('baseline_metrics')}")
            print(temporal.get("warning", ""))
        print("Top features:")
        for row in model.get("feature_importance", [])[:10]:
            print(f"  {row['feature']}: {row['importance']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default="hit_in_game")
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Skip the latest-season time-held-out estimator check.",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    df = pd.read_parquet(args.dataset)

    artifact = {}
    if args.model and os.path.exists(args.model):
        loaded = joblib.load(args.model)
        if isinstance(loaded, dict):
            artifact = loaded

    features = list(artifact.get("features", []))
    report = audit_dataframe(
        df,
        target=args.target if args.target in df.columns else None,
        feature_columns=features or None,
    )
    if artifact:
        report["model"] = model_audit(df, artifact, target=args.target)
        if not args.no_temporal:
            report["temporal_validation"] = temporal_holdout_audit(
                df, artifact, target=args.target
            )

    print_report(report)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=_json_value)
        print(f"Wrote JSON audit: {args.json_out}")


if __name__ == "__main__":
    main()
