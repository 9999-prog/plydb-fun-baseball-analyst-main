import unittest

import pandas as pd

from build_advanced_matchup_features import build_advanced_feature_table
from modern_stats import blend_rate
from prop_metrics import (
    best_total_market,
    projected_nrfi_prob,
    projected_score_profile,
    prop_reason_under,
    recent_team_scoring_profile,
    team_first_inning_runs,
    totals_pick,
)


class PropMetricTests(unittest.TestCase):
    def setUp(self):
        self.pa = pd.DataFrame(
            [
                {"game_pk": 1, "game_date": "2026-08-01", "batting_team": "AAA", "pitching_team": "BBB", "inning": 1, "runs_on_pa": 1},
                {"game_pk": 1, "game_date": "2026-08-01", "batting_team": "AAA", "pitching_team": "BBB", "inning": 2, "runs_on_pa": 2},
                {"game_pk": 1, "game_date": "2026-08-01", "batting_team": "BBB", "pitching_team": "AAA", "inning": 1, "runs_on_pa": 0},
                {"game_pk": 2, "game_date": "2026-08-02", "batting_team": "AAA", "pitching_team": "BBB", "inning": 1, "runs_on_pa": 3},
                {"game_pk": 2, "game_date": "2026-08-02", "batting_team": "BBB", "pitching_team": "AAA", "inning": 1, "runs_on_pa": 1},
            ]
        )
        self.pa["game_date"] = pd.to_datetime(self.pa["game_date"])

    def test_run_aggregates_use_runs_on_pa_column(self):
        self.assertEqual(team_first_inning_runs("AAA", self.pa, "2026-08-03"), 2.0)
        profile = recent_team_scoring_profile("AAA", self.pa, "2026-08-03")
        self.assertEqual(profile["scored"], 3.0)
        self.assertEqual(profile["allowed"], 0.5)
        self.assertEqual(profile["scored_games"], 2)
        self.assertEqual(profile["allowed_games"], 2)

    def test_neutral_score_profile_sums_two_team_scores(self):
        profile = projected_score_profile(
            "AAA", "BBB", pd.DataFrame(), "2026-08-03", modern_stats={}
        )
        self.assertAlmostEqual(profile["projected_home_runs"], 4.2)
        self.assertAlmostEqual(profile["projected_away_runs"], 4.2)
        self.assertAlmostEqual(profile["projected_total"], 8.4)
        self.assertAlmostEqual(profile["projected_margin_home"], 0.0)

    def test_total_reason_follows_over_direction(self):
        reason = prop_reason_under({
            "projected_total": 10.0,
            "market_line": 8.5,
            "model_over_prob": 0.64,
            "model_under_prob": 0.36,
            "data_coverage": 1.0,
            "sample_games": 12,
        })
        self.assertIn("over", reason.lower())
        self.assertNotIn("below the market", reason.lower())

    def test_totals_reuses_supplied_score_profile(self):
        result = totals_pick(
            "BBB", "AAA", self.pa, "2026-08-03", [], modern_stats={},
            score_profile={
                "projected_home_runs": 5.0,
                "projected_away_runs": 4.0,
                "projected_total": 9.0,
                "projected_margin_home": 1.0,
                "coverage": 1.0,
                "sample_games": 20,
                "modern_weight": 0.7,
            },
        )
        self.assertAlmostEqual(result["projected_total"], 9.0)

    def test_total_market_filters_extreme_price_and_matches_aliases(self):
        odds = [{
            "home_team": "Los Angeles Dodgers",
            "away_team": "New York Yankees",
            "bookmakers": [
                {"markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -110},
                    {"name": "Under", "point": 8.5, "price": -110},
                ]}]},
                {"markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -105},
                    {"name": "Under", "point": 8.5, "price": -115},
                ]}]},
                {"markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "point": 8.5, "price": 2500},
                    {"name": "Under", "point": 8.5, "price": -800},
                ]}]},
            ],
        }]
        over = best_total_market("LAD", "NYY", odds, side="over")
        self.assertEqual(over["point"], 8.5)
        self.assertNotEqual(over["price"], 2500)

    def test_missing_current_window_is_neutral_and_passes(self):
        odds = [
            {
                "home_team": "BBB",
                "away_team": "AAA",
                "bookmakers": [
                    {
                        "markets": [
                            {"key": "totals", "outcomes": [
                                {"name": "Over", "point": 8.5, "price": -110},
                                {"name": "Under", "point": 8.5, "price": -110},
                            ]}
                        ]
                    }
                ],
            }
        ]
        self.assertEqual(projected_nrfi_prob("BBB", "AAA", self.pa, "2027-08-03"), 0.5)
        result = totals_pick(
            "BBB", "AAA", self.pa, "2027-08-03", odds, modern_stats={}
        )
        self.assertEqual(result["data_coverage"], 0.0)
        self.assertEqual(result["model_under_prob"], 0.5)
        self.assertEqual(result["pick"], "PASS")

    def test_modern_stats_are_used_when_supplied(self):
        modern = {
            "AAA": {
                "runs_per_game": 5.0,
                "runs_allowed_per_game": 3.0,
                "modern_games": 20,
            },
            "BBB": {
                "runs_per_game": 3.0,
                "runs_allowed_per_game": 5.0,
                "modern_games": 20,
            },
        }
        result = totals_pick(
            "BBB", "AAA", self.pa, "2026-08-03", [], modern_stats=modern
        )
        self.assertAlmostEqual(result["modern_weight"], 0.7)
        self.assertEqual(result["data_coverage"], 1.0)


class ModernBlendTests(unittest.TestCase):
    def test_full_sample_is_seventy_thirty(self):
        value, weight = blend_rate(4.0, 20, 6.0, 20)
        self.assertAlmostEqual(weight, 0.7)
        self.assertAlmostEqual(value, 5.4)

    def test_one_game_does_not_dominate(self):
        value, weight = blend_rate(4.0, 20, 6.0, 1)
        self.assertAlmostEqual(weight, 0.07)
        self.assertAlmostEqual(value, 4.14)


class AdvancedFeatureTests(unittest.TestCase):
    def test_missing_optional_batting_features_do_not_crash(self):
        pa = pd.DataFrame(
            [
                {
                    "game_pk": 10,
                    "game_date": "2026-08-01",
                    "home_team": "HOM",
                    "away_team": "AWY",
                    "inning_topbot": "Top",
                    "inning": 1,
                    "at_bat_number": 1,
                    "batter": 101,
                    "pitcher": 201,
                    "p_throws": "R",
                    "events": "single",
                    "batter_roll_hit_rate": 0.25,
                    "batter_roll_xba": 0.22,
                    "pitcher_roll_k_rate": 0.24,
                    "pitcher_roll_bb_rate": 0.08,
                    "pitcher_roll_xba_against": 0.23,
                    "pitcher_roll_hardhit_against": 0.35,
                    "pitcher_roll_velo": 94.0,
                    "pitcher_roll_spin": 2200.0,
                    "park_hit_factor": 1.02,
                    "park_hr_factor": 0.98,
                },
                {
                    "game_pk": 10,
                    "game_date": "2026-08-01",
                    "home_team": "HOM",
                    "away_team": "AWY",
                    "inning_topbot": "Bot",
                    "inning": 1,
                    "at_bat_number": 2,
                    "batter": 102,
                    "pitcher": 202,
                    "p_throws": "L",
                    "events": "field_out",
                    "batter_roll_hit_rate": 0.30,
                    "batter_roll_xba": 0.25,
                    "pitcher_roll_k_rate": 0.21,
                    "pitcher_roll_bb_rate": 0.09,
                    "pitcher_roll_xba_against": 0.24,
                    "pitcher_roll_hardhit_against": 0.37,
                    "pitcher_roll_velo": 93.0,
                    "pitcher_roll_spin": 2150.0,
                    "park_hit_factor": 1.02,
                    "park_hr_factor": 0.98,
                },
            ]
        )
        result = build_advanced_feature_table(pa)
        self.assertEqual(len(result), 1)
        self.assertIn("home_lineup_xwoba", result.columns)
        self.assertIn("away_lineup_hr_rate", result.columns)
        self.assertEqual(result.loc[0, "home_starter_id"], 201)
        self.assertEqual(result.loc[0, "away_starter_id"], 202)

    def test_score_deltas_feed_bullpen_run_rate(self):
        pa = pd.DataFrame(
            [
                {
                    "game_pk": 11, "game_date": "2026-08-01",
                    "home_team": "HOM", "away_team": "AWY",
                    "inning_topbot": "Top", "inning": 1,
                    "at_bat_number": 1, "batter": 111, "pitcher": 211,
                    "p_throws": "R", "events": "field_out",
                    "bat_score": 0, "post_bat_score": 0,
                },
                {
                    "game_pk": 11, "game_date": "2026-08-01",
                    "home_team": "HOM", "away_team": "AWY",
                    "inning_topbot": "Top", "inning": 1,
                    "at_bat_number": 2, "batter": 112, "pitcher": 213,
                    "p_throws": "R", "events": "single",
                    "bat_score": 0, "post_bat_score": 1,
                },
                {
                    "game_pk": 11, "game_date": "2026-08-01",
                    "home_team": "HOM", "away_team": "AWY",
                    "inning_topbot": "Bot", "inning": 1,
                    "at_bat_number": 3, "batter": 113, "pitcher": 212,
                    "p_throws": "L", "events": "field_out",
                    "bat_score": 0, "post_bat_score": 0,
                },
            ]
        )
        result = build_advanced_feature_table(pa)
        self.assertAlmostEqual(float(result.loc[0, "home_bullpen_rpa"]), 1.0)


if __name__ == "__main__":
    unittest.main()
