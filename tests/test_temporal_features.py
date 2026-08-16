import unittest

import pandas as pd

from temporal_features import (
    StaticSnapshotAdapter,
    TemporalDataError,
    TimestampedSnapshotAdapter,
    as_of_filter,
    latest_entity_snapshot,
)


class TemporalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = "2025-04-07T17:00:00Z"
        self.records = pd.DataFrame(
            [
                {
                    "batter": 10, "value": 0.20, "source": "statcast",
                    "event_at": "2025-04-07T12:00:00Z",
                    "available_at": "2025-04-07T12:05:00Z",
                    "ingested_at": "2025-04-07T12:06:00Z", "revision": "1",
                },
                {
                    "batter": 10, "value": 0.25, "source": "statcast",
                    "event_at": "2025-04-07T15:00:00Z",
                    "available_at": "2025-04-07T15:05:00Z",
                    "ingested_at": "2025-04-07T15:06:00Z", "revision": "1",
                },
                {
                    "batter": 10, "value": 0.90, "source": "statcast",
                    "event_at": "2025-04-07T18:00:00Z",
                    "available_at": "2025-04-07T18:05:00Z",
                    "ingested_at": "2025-04-07T18:06:00Z", "revision": "1",
                },
            ]
        )

    def test_before_exactly_at_and_after_cutoff(self):
        frame = pd.DataFrame([
            {"batter": 1, "available_at": "2025-01-01T11:59:59Z"},
            {"batter": 2, "available_at": "2025-01-01T12:00:00Z"},
            {"batter": 3, "available_at": "2025-01-01T12:00:01Z"},
        ])
        result = as_of_filter(frame, "2025-01-01T12:00:00Z")
        self.assertEqual(result["batter"].tolist(), [1])

    def test_same_day_information_before_cutoff_is_eligible(self):
        result = latest_entity_snapshot(
            self.records, entity_columns=["batter"], prediction_cutoff=self.cutoff,
        )
        self.assertEqual(result.loc[0, "value"], 0.25)
        self.assertEqual(str(result.loc[0, "prediction_cutoff"]), "2025-04-07 17:00:00+00:00")

    def test_timezone_and_daylight_saving_boundaries_are_compared_in_utc(self):
        # The two 01:30 local times on US fall-back day are distinct instants.
        frame = pd.DataFrame([
            {"batter": 1, "available_at": "2025-11-02T01:30:00-04:00"},  # 05:30Z
            {"batter": 2, "available_at": "2025-11-02T01:30:00-05:00"},  # 06:30Z
        ])
        result = as_of_filter(frame, "2025-11-02T06:00:00Z")
        self.assertEqual(result["batter"].tolist(), [1])
        self.assertEqual(str(result.loc[result.index[0], "available_at"]), "2025-11-02 05:30:00+00:00")

    def test_late_published_revision_is_ineligible_until_available(self):
        frame = pd.DataFrame([
            {
                "batter": 10, "value": 0.20, "source": "source-a", "revision": "1",
                "event_at": "2025-06-01T12:00:00Z", "available_at": "2025-06-01T12:01:00Z",
                "ingested_at": "2025-06-01T12:02:00Z",
            },
            {
                "batter": 10, "value": 0.30, "source": "source-a", "revision": "2",
                "event_at": "2025-06-01T12:00:00Z", "available_at": "2025-06-02T09:00:00Z",
                "ingested_at": "2025-06-02T09:01:00Z",
            },
        ])
        early = latest_entity_snapshot(
            frame, entity_columns=["batter"], prediction_cutoff="2025-06-01T18:00:00Z",
        )
        late = latest_entity_snapshot(
            frame, entity_columns=["batter"], prediction_cutoff="2025-06-02T10:00:00Z",
        )
        self.assertEqual(early.loc[0, "value"], 0.20)
        self.assertEqual(late.loc[0, "value"], 0.30)
        self.assertEqual(late.loc[0, "revision"], "2")

    def test_duplicates_and_shuffled_input_have_deterministic_tie_breaking(self):
        frame = pd.DataFrame([
            {"batter": 10, "value": "A", "source": "alpha", "revision": "1", "available_at": "2025-01-01T01:00:00Z"},
            {"batter": 10, "value": "B", "source": "beta", "revision": "1", "available_at": "2025-01-01T01:00:00Z"},
            {"batter": 10, "value": "B", "source": "beta", "revision": "1", "available_at": "2025-01-01T01:00:00Z"},
        ])
        first = latest_entity_snapshot(
            frame, entity_columns=["batter"], prediction_cutoff="2025-01-02T00:00:00Z",
        )
        second = latest_entity_snapshot(
            frame.sample(frac=1, random_state=7), entity_columns=["batter"],
            prediction_cutoff="2025-01-02T00:00:00Z",
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.loc[0, "value"], "B")

    def test_unknown_entities_and_empty_snapshots(self):
        adapter = TimestampedSnapshotAdapter(self.records, ("batter",))
        unknown = adapter.snapshot(self.cutoff, entity_filter={"batter": [999]})
        self.assertTrue(unknown.empty)
        empty = TimestampedSnapshotAdapter(self.records.iloc[0:0], ("batter",)).snapshot(self.cutoff)
        self.assertTrue(empty.empty)

    def test_missing_malformed_naive_and_ambiguous_timestamps_fail_closed(self):
        cases = [
            pd.DataFrame([{ "batter": 1 }]),
            pd.DataFrame([{ "batter": 1, "available_at": "not-a-time" }]),
            pd.DataFrame([{ "batter": 1, "available_at": "2025-01-01 12:00:00" }]),
            pd.DataFrame([{ "batter": 1, "available_at": "2025-11-02 01:30:00" }]),
            pd.DataFrame([{ "batter": 1, "available_at": "2025-01-01T12:00:00Z", "event_at": "2025-01-01 10:00:00" }]),
        ]
        for frame in cases:
            with self.subTest(frame=frame.to_dict("records")):
                with self.assertRaises(TemporalDataError):
                    as_of_filter(frame, "2025-01-02T00:00:00Z")
        with self.assertRaises(TemporalDataError):
            as_of_filter(pd.DataFrame([{ "batter": 1, "available_at": "2025-01-01T12:00:00Z" }]), "2025-01-02 00:00:00")

    def test_no_mutation_and_metadata_are_preserved(self):
        before = self.records.copy(deep=True)
        result = latest_entity_snapshot(
            self.records, entity_columns=["batter"], prediction_cutoff=self.cutoff,
        )
        pd.testing.assert_frame_equal(self.records, before)
        self.assertEqual(
            set(("source", "event_at", "available_at", "ingested_at", "revision", "batter", "prediction_cutoff")),
            set(("source", "event_at", "available_at", "ingested_at", "revision", "batter", "prediction_cutoff")).intersection(result.columns),
        )

    def test_static_adapter_is_explicit_rollback_only(self):
        original = pd.DataFrame([{ "batter": 10, "value": 0.30 }])
        result = StaticSnapshotAdapter({"batter": original}).snapshot(
            "batter", "2025-01-01T12:00:00Z",
        )
        self.assertEqual(result.loc[0, "snapshot_provenance"], "legacy_static_rollback")
        self.assertEqual(str(result.loc[0, "prediction_cutoff"]), "2025-01-01 12:00:00+00:00")
        self.assertNotIn("prediction_cutoff", original.columns)


if __name__ == "__main__":
    unittest.main()
