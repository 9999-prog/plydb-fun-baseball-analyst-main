import unittest

import pandas as pd

from temporal_features import (
    TemporalDataError,
    as_of_filter,
    latest_entity_snapshot,
)


class TemporalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame(
            [
                {"game_date": "2025-04-01", "game_pk": 1, "batter": 10, "value": 0.20},
                {"game_date": "2025-04-05", "game_pk": 2, "batter": 10, "value": 0.25},
                # Must never be visible to an April 6 pregame prediction.
                {"game_date": "2025-04-07", "game_pk": 3, "batter": 10, "value": 0.90},
                {"game_date": "2025-04-07", "game_pk": 3, "batter": 20, "value": 0.80},
            ]
        )

    def test_as_of_filter_excludes_future_and_same_day_rows(self):
        result = as_of_filter(self.rows, "2025-04-07T00:00:00Z")
        self.assertEqual(result["game_pk"].tolist(), [1, 2])

    def test_as_of_filter_does_not_mutate_input(self):
        original_dates = self.rows["game_date"].tolist()
        as_of_filter(self.rows, "2025-04-07T00:00:00Z")
        self.assertEqual(self.rows["game_date"].tolist(), original_dates)

    def test_latest_snapshot_uses_only_pre_cutoff_history(self):
        snapshot = latest_entity_snapshot(
            self.rows,
            entity_columns=["batter"],
            as_of_timestamp="2025-04-07T00:00:00Z",
            value_columns=["value"],
        )
        player_10 = snapshot.loc[snapshot["batter"] == 10].iloc[0]
        self.assertEqual(player_10["game_pk"], 2)
        self.assertAlmostEqual(player_10["value"], 0.25)
        self.assertNotIn(20, snapshot["batter"].tolist())

    def test_missing_time_column_fails_closed(self):
        with self.assertRaises(TemporalDataError):
            as_of_filter(self.rows.drop(columns="game_date"), "2025-04-07T00:00:00Z")

    def test_invalid_timestamp_fails_closed(self):
        with self.assertRaises(TemporalDataError):
            as_of_filter(self.rows, "not-a-timestamp")


if __name__ == "__main__":
    unittest.main()
