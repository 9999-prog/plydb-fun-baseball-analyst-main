import unittest

import pandas as pd

from advanced_metrics import (
    batter_form_metrics,
    batter_pitcher_h2h_metrics,
    build_team_metric_cards,
    expected_value_per_unit,
    market_card,
    modern_batter_context,
    shrink_rate,
    team_h2h_metrics,
)


class AdvancedMetricTests(unittest.TestCase):
    def test_small_samples_shrink_toward_neutral(self):
        self.assertAlmostEqual(shrink_rate(0.0, 1, 0.57, 5), 0.475)
        self.assertAlmostEqual(shrink_rate(1.0, 1, 0.57, 5), 0.6416666667)

    def _batter_pa(self):
        rows = []
        for game_number in range(1, 13):
            date = pd.Timestamp("2026-07-01") + pd.to_timedelta(game_number, unit="D")
            # Two PA per game; every third game has a hit.
            for pa_number in range(2):
                rows.append({
                    "game_pk": game_number,
                    "game_date": date,
                    "batter": 10,
                    "pitcher": 20,
                    "is_hit": int(game_number % 3 == 0 and pa_number == 0),
                    "events": "single" if game_number % 3 == 0 and pa_number == 0 else "field_out",
                    "estimated_ba_using_speedangle": 0.30 if game_number % 3 == 0 else 0.20,
                    "estimated_woba_using_speedangle": 0.38 if game_number % 3 == 0 else 0.25,
                    "launch_speed": 100 if game_number % 3 == 0 else 85,
                })
        # This future hit must never enter a pregame card.
        rows.append({
            "game_pk": 999, "game_date": pd.Timestamp("2026-09-01"),
            "batter": 10, "pitcher": 20, "is_hit": 1, "events": "home_run",
            "estimated_ba_using_speedangle": 0.50,
            "estimated_woba_using_speedangle": 0.60,
            "launch_speed": 110,
        })
        return pd.DataFrame(rows)

    def test_form_windows_are_game_level_and_exclude_future(self):
        pa = self._batter_pa()
        card = batter_form_metrics(pa, 10, "2026-08-01")
        self.assertEqual(card["last5_games"], 5)
        self.assertEqual(card["last10_games"], 10)
        self.assertEqual(card["last10_pa"], 20)
        self.assertNotEqual(card["last10_games"], 11)
        self.assertEqual(card["data_status"], "SUPPORTED")
        self.assertIsNotNone(card["minimum_credible_hit_rate"])

    def test_h2h_uses_directional_real_scores(self):
        rows = []
        outcomes = [1, 0, 1, 1]
        for game_number, home_won in enumerate(outcomes, start=1):
            home_score, away_score = (5, 2) if home_won else (2, 5)
            date = pd.Timestamp("2026-06-01") + pd.to_timedelta(game_number, unit="D")
            rows.extend([
                {
                    "game_pk": game_number, "game_date": date,
                    "home_team": "AAA", "away_team": "BBB",
                    "batting_team": "AAA", "pitching_team": "BBB",
                    "runs_on_pa": home_score, "post_home_score": home_score,
                    "post_away_score": away_score,
                },
                {
                    "game_pk": game_number, "game_date": date,
                    "home_team": "AAA", "away_team": "BBB",
                    "batting_team": "BBB", "pitching_team": "AAA",
                    "runs_on_pa": away_score, "post_home_score": home_score,
                    "post_away_score": away_score,
                },
            ])
        result = team_h2h_metrics(pd.DataFrame(rows), "AAA", "BBB", "2026-07-01")
        self.assertEqual(result["h2h_games"], 4)
        self.assertGreater(result["h2h_l5"], 0.5)
        self.assertGreater(result["h2h_delta"], 0.0)

    def test_batter_pitcher_h2h_requires_actual_matchups(self):
        pa = self._batter_pa()
        result = batter_pitcher_h2h_metrics(pa, 10, 20, "2026-08-01")
        self.assertEqual(result["h2h_games"], 12)
        self.assertEqual(result["h2h_pa"], 24)
        self.assertEqual(result["h2h_status"], "SUPPORTED")
        missing = batter_pitcher_h2h_metrics(pa, 999, 20, "2026-08-01")
        self.assertIsNone(missing["h2h_l10"])
        self.assertEqual(missing["h2h_status"], "NO_DATA")

    def test_market_aliases_and_outlier_prices_are_safe(self):
        odds = [{
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "New York Yankees", "price": -110},
                    {"name": "Boston Red Sox", "price": -110},
                ]}]},
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "New York Yankees", "price": -105},
                    {"name": "Boston Red Sox", "price": -115},
                ]}]},
                {"markets": [{"key": "h2h", "outcomes": [
                    {"name": "New York Yankees", "price": 2500},
                    {"name": "Boston Red Sox", "price": -800},
                ]}]},
            ],
        }]
        market = market_card(odds, "NYY", "BOS")
        self.assertTrue(market["available"])
        self.assertEqual(market["price_filter"], "ROBUST")
        self.assertGreaterEqual(market["outlier_count"], 1)
        self.assertNotEqual(market["home_best_price"], 2500)

    def test_market_is_no_vig_and_ev_uses_price(self):
        odds = [{
            "home_team": "AAA", "away_team": "BBB",
            "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
                {"name": "AAA", "price": -110},
                {"name": "BBB", "price": 100},
            ]}]}],
        }]
        market = market_card(odds, "AAA", "BBB")
        self.assertTrue(market["available"])
        self.assertAlmostEqual(
            market["home_market_probability"] + market["away_market_probability"],
            1.0,
        )
        self.assertAlmostEqual(expected_value_per_unit(0.60, -110), 0.1454545454)

    def test_modern_player_context_is_labelled_proxy(self):
        context = modern_batter_context(
            {"games_played": 20, "plate_appearances": 80, "hits": 24},
            usage_pa_per_game=4.0,
        )
        self.assertTrue(context["available"])
        self.assertGreater(context["modern_game_hit_proxy"], context["pa_hit_rate"])
        self.assertIn("not a game-hit observation", context["proxy_note"])

    def test_team_card_returns_n_a_ev_without_market(self):
        result = {
            "home_team": "AAA", "away_team": "BBB",
            "home_win_prob": 0.60, "away_win_prob": 0.40,
            "signal_quality": 0.8,
            "modern_games_home": 20, "modern_games_away": 20,
        }
        cards = build_team_metric_cards(
            result,
            score_profile={
                "projected_total": 8.5,
                "projected_home_runs": 4.6,
                "projected_away_runs": 3.9,
                "projected_margin_home": 0.7,
            },
            h2h={"h2h_delta": 0.05, "h2h_l5": 0.6, "h2h_l10": 0.55, "h2h_games": 8},
            odds_data=[],
        )
        self.assertIsNone(cards["home"]["expected_value"])
        self.assertAlmostEqual(cards["home"]["pred_total"], 8.5)
        self.assertAlmostEqual(cards["away"]["avg_margin"], -0.7)


if __name__ == "__main__":
    unittest.main()
