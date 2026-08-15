"""
MLB PREDICTION CACHE ADAPTER

Reads the historical prediction caches created by:
    build_prediction_caches.py

The prediction engine should use this module instead of treating
the cache files as optional decoration.

Caches:
    team_season_stats.joblib
    team_recent_stats.joblib
    h2h_stats.joblib
    pitcher_matchups.joblib
    batter_matchups.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd


class PredictionCache:

    def __init__(self, root):

        self.root = root

        self.team_season = self._load(
            "team_season_stats.joblib"
        )

        self.team_recent = self._load(
            "team_recent_stats.joblib"
        )

        self.h2h = self._load(
            "h2h_stats.joblib"
        )

        self.pitcher_matchups = self._load(
            "pitcher_matchups.joblib"
        )

        self.batter_matchups = self._load(
            "batter_matchups.joblib"
        )

        print("\nCACHE STATUS")
        print("=" * 70)

        self._status(
            "team_season_stats",
            self.team_season
        )

        self._status(
            "team_recent_stats",
            self.team_recent
        )

        self._status(
            "h2h_stats",
            self.h2h
        )

        self._status(
            "pitcher_matchups",
            self.pitcher_matchups
        )

        self._status(
            "batter_matchups",
            self.batter_matchups
        )

        print("=" * 70)


    # ============================================================
    # LOAD
    # ============================================================

    def _load(self, filename):

        path = os.path.join(
            self.root,
            filename
        )

        if not os.path.exists(path):

            print(
                f"WARNING: {filename} not found"
            )

            return None

        try:

            return joblib.load(path)

        except Exception as exc:

            print(
                f"WARNING: failed loading "
                f"{filename}: {exc}"
            )

            return None


    # ============================================================
    # STATUS
    # ============================================================

    def _status(self, name, data):

        if data is None:

            print(
                f"  {name}: MISSING"
            )

            return

        if isinstance(data, pd.DataFrame):

            print(
                f"  {name}: "
                f"{len(data):,} rows"
            )

        elif isinstance(data, dict):

            print(
                f"  {name}: "
                f"{len(data):,} records"
            )

        else:

            print(
                f"  {name}: loaded"
            )


    # ============================================================
    # NUMERIC
    # ============================================================

    @staticmethod
    def num(value):

        try:

            if value is None:
                return np.nan

            value = float(value)

            if np.isfinite(value):
                return value

        except Exception:
            pass

        return np.nan


    # ============================================================
    # FIND TEAM ROW
    # ============================================================

    def _team_row(self, data, team):

        if data is None:
            return None

        if isinstance(data, dict):

            value = data.get(team)

            if isinstance(value, dict):
                return value

            if isinstance(value, pd.Series):
                return value

        if isinstance(data, pd.DataFrame):

            team_columns = [
                "team",
                "team_code",
                "abbreviation",
                "team_abbreviation",
            ]

            for column in team_columns:

                if column not in data.columns:
                    continue

                rows = data[
                    data[column].astype(str).str.upper()
                    == str(team).upper()
                ]

                if not rows.empty:

                    return rows.iloc[0]

        return None


    # ============================================================
    # GENERIC FIELD
    # ============================================================

    def _field(
        self,
        row,
        names,
    ):

        if row is None:
            return np.nan

        if isinstance(row, pd.Series):

            for name in names:

                if name in row.index:

                    value = self.num(
                        row[name]
                    )

                    if not np.isnan(value):
                        return value

        if isinstance(row, dict):

            for name in names:

                if name in row:

                    value = self.num(
                        row[name]
                    )

                    if not np.isnan(value):
                        return value

        return np.nan


    # ============================================================
    # TEAM SEASON
    # ============================================================

    def team_season_stats(self, team):

        row = self._team_row(
            self.team_season,
            team
        )

        if row is None:
            return {}

        return {
            "games":
                self._field(
                    row,
                    [
                        "games",
                        "season_games",
                        "game_count",
                    ],
                ),

            "hits":
                self._field(
                    row,
                    [
                        "hits",
                        "season_hits",
                    ],
                ),

            "home_runs":
                self._field(
                    row,
                    [
                        "home_runs",
                        "hr",
                        "season_hr",
                    ],
                ),

            "walks":
                self._field(
                    row,
                    [
                        "walks",
                        "bb",
                        "season_bb",
                    ],
                ),

            "strikeouts":
                self._field(
                    row,
                    [
                        "strikeouts",
                        "so",
                        "season_so",
                    ],
                ),

            "hit_rate":
                self._field(
                    row,
                    [
                        "hit_rate",
                        "season_hit_rate",
                    ],
                ),

            "xba":
                self._field(
                    row,
                    [
                        "xBA",
                        "xba",
                        "season_xba",
                    ],
                ),

            "xwoba":
                self._field(
                    row,
                    [
                        "xwOBA",
                        "xwoba",
                        "season_xwoba",
                    ],
                ),

            "exit_velocity":
                self._field(
                    row,
                    [
                        "exit_velocity",
                        "ev",
                        "season_ev",
                    ],
                ),

            "hard_hit":
                self._field(
                    row,
                    [
                        "hard_hit_rate",
                        "hard_hit",
                        "hardhit",
                        "season_hard_hit",
                    ],
                ),

            "opponent_hit_rate":
                self._field(
                    row,
                    [
                        "opponent_hit_rate",
                        "opp_hit_rate",
                    ],
                ),

            "opponent_xba":
                self._field(
                    row,
                    [
                        "opponent_xba",
                        "opp_xba",
                    ],
                ),

            "opponent_xwoba":
                self._field(
                    row,
                    [
                        "opponent_xwoba",
                        "opp_xwoba",
                    ],
                ),

            "opponent_exit_velocity":
                self._field(
                    row,
                    [
                        "opponent_exit_velocity",
                        "opp_ev",
                    ],
                ),

            "opponent_hard_hit":
                self._field(
                    row,
                    [
                        "opponent_hard_hit",
                        "opp_hard_hit",
                    ],
                ),
        }


    # ============================================================
    # TEAM RECENT
    # ============================================================

    def team_recent_stats(self, team):

        row = self._team_row(
            self.team_recent,
            team
        )

        if row is None:
            return {}

        return {
            "last5_hit_rate":
                self._field(
                    row,
                    [
                        "last5_hit_rate",
                        "hit_rate_last5",
                        "last_5_hit_rate",
                    ],
                ),

            "last10_hit_rate":
                self._field(
                    row,
                    [
                        "last10_hit_rate",
                        "hit_rate_last10",
                        "last_10_hit_rate",
                    ],
                ),

            "last5_xba":
                self._field(
                    row,
                    [
                        "last5_xba",
                        "xBA_last5",
                        "last_5_xba",
                    ],
                ),

            "last10_xba":
                self._field(
                    row,
                    [
                        "last10_xba",
                        "xBA_last10",
                        "last_10_xba",
                    ],
                ),

            "last5_xwoba":
                self._field(
                    row,
                    [
                        "last5_xwoba",
                        "xwOBA_last5",
                    ],
                ),

            "last10_xwoba":
                self._field(
                    row,
                    [
                        "last10_xwoba",
                        "xwOBA_last10",
                    ],
                ),

            "last5_ev":
                self._field(
                    row,
                    [
                        "last5_exit_velocity",
                        "last5_ev",
                        "ev_last5",
                    ],
                ),

            "last10_ev":
                self._field(
                    row,
                    [
                        "last10_exit_velocity",
                        "last10_ev",
                        "ev_last10",
                    ],
                ),

            "last5_hard_hit":
                self._field(
                    row,
                    [
                        "last5_hard_hit_rate",
                        "last5_hard_hit",
                        "hard_hit_last5",
                    ],
                ),

            "last10_hard_hit":
                self._field(
                    row,
                    [
                        "last10_hard_hit_rate",
                        "last10_hard_hit",
                        "hard_hit_last10",
                    ],
                ),

            "opponent_hit_rate":
                self._field(
                    row,
                    [
                        "opponent_hit_rate",
                        "opp_hit_rate",
                    ],
                ),

            "opponent_xba":
                self._field(
                    row,
                    [
                        "opponent_xba",
                        "opp_xba",
                    ],
                ),

            "opponent_xwoba":
                self._field(
                    row,
                    [
                        "opponent_xwoba",
                        "opp_xwoba",
                    ],
                ),
        }


    # ============================================================
    # H2H
    # ============================================================

    def h2h_stats_for(
        self,
        team,
        opponent
    ):

        data = self.h2h

        if data is None:
            return {}

        if isinstance(data, pd.DataFrame):

            cols = set(data.columns)

            team_col = next(
                (
                    c for c in [
                        "team",
                        "team_code",
                        "team_a",
                    ]
                    if c in cols
                ),
                None
            )

            opp_col = next(
                (
                    c for c in [
                        "opponent",
                        "opponent_team",
                        "team_b",
                    ]
                    if c in cols
                ),
                None
            )

            if team_col and opp_col:

                rows = data[
                    (
                        data[team_col].astype(str).str.upper()
                        == team.upper()
                    )
                    &
                    (
                        data[opp_col].astype(str).str.upper()
                        == opponent.upper()
                    )
                ]

                if not rows.empty:

                    row = rows.iloc[-1]

                    return {
                        "games":
                            self._field(
                                row,
                                [
                                    "games",
                                    "game_count",
                                    "h2h_games",
                                ],
                            ),

                        "wins":
                            self._field(
                                row,
                                [
                                    "wins",
                                    "team_wins",
                                    "h2h_wins",
                                ],
                            ),

                        "win_rate":
                            self._field(
                                row,
                                [
                                    "win_rate",
                                    "h2h_win_rate",
                                ],
                            ),

                        "hits":
                            self._field(
                                row,
                                [
                                    "hits",
                                    "h2h_hits",
                                ],
                            ),

                        "runs":
                            self._field(
                                row,
                                [
                                    "runs",
                                    "h2h_runs",
                                ],
                            ),
                    }

        return {}


    # ============================================================
    # PITCHER MATCHUP
    # ============================================================

    def pitcher_matchup(
        self,
        pitcher_id,
        opponent
    ):

        data = self.pitcher_matchups

        if data is None:
            return {}

        if isinstance(data, pd.DataFrame):

            rows = data.copy()

            pitcher_cols = [
                "pitcher",
                "pitcher_id",
                "mlbam_pitcher_id",
            ]

            pitcher_col = next(
                (
                    c for c in pitcher_cols
                    if c in rows.columns
                ),
                None
            )

            opponent_cols = [
                "opponent",
                "opponent_team",
                "batting_team",
            ]

            opponent_col = next(
                (
                    c for c in opponent_cols
                    if c in rows.columns
                ),
                None
            )

            if pitcher_col:

                rows = rows[
                    rows[pitcher_col].astype(str)
                    == str(pitcher_id)
                ]

            if opponent_col:

                rows = rows[
                    rows[opponent_col].astype(str).str.upper()
                    == opponent.upper()
                ]

            if not rows.empty:

                row = rows.iloc[-1]

                return {
                    "games":
                        self._field(
                            row,
                            [
                                "games",
                                "matchup_games",
                                "appearances",
                            ],
                        ),

                    "k_rate":
                        self._field(
                            row,
                            [
                                "k_rate",
                                "strikeout_rate",
                            ],
                        ),

                    "bb_rate":
                        self._field(
                            row,
                            [
                                "bb_rate",
                                "walk_rate",
                            ],
                        ),

                    "xba":
                        self._field(
                            row,
                            [
                                "xba",
                                "xBA",
                                "expected_ba",
                            ],
                        ),

                    "xwoba":
                        self._field(
                            row,
                            [
                                "xwoba",
                                "xwOBA",
                                "expected_woba",
                            ],
                        ),
                }

        return {}


    # ============================================================
    # BATTER VS PITCHER
    # ============================================================

    def batter_matchup(
        self,
        batter_id,
        pitcher_id
    ):

        data = self.batter_matchups

        if data is None:
            return {}

        if isinstance(data, pd.DataFrame):

            rows = data.copy()

            batter_col = next(
                (
                    c for c in [
                        "batter",
                        "batter_id",
                        "mlbam_batter_id",
                    ]
                    if c in rows.columns
                ),
                None
            )

            pitcher_col = next(
                (
                    c for c in [
                        "pitcher",
                        "pitcher_id",
                        "mlbam_pitcher_id",
                    ]
                    if c in rows.columns
                ),
                None
            )

            if batter_col:

                rows = rows[
                    rows[batter_col].astype(str)
                    == str(batter_id)
                ]

            if pitcher_col:

                rows = rows[
                    rows[pitcher_col].astype(str)
                    == str(pitcher_id)
                ]

            if not rows.empty:

                row = rows.iloc[-1]

                return {
                    "pa":
                        self._field(
                            row,
                            [
                                "pa",
                                "plate_appearances",
                                "matchup_pa",
                            ],
                        ),

                    "hits":
                        self._field(
                            row,
                            [
                                "hits",
                                "matchup_hits",
                            ],
                        ),

                    "hit_rate":
                        self._field(
                            row,
                            [
                                "hit_rate",
                                "batting_average",
                            ],
                        ),

                    "xba":
                        self._field(
                            row,
                            [
                                "xba",
                                "xBA",
                            ],
                        ),

                    "xwoba":
                        self._field(
                            row,
                            [
                                "xwoba",
                                "xwOBA",
                            ],
                        ),
                }

        return {}