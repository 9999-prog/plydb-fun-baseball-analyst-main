import io
import unittest
from contextlib import redirect_stdout

from speech_output import build_narration, speak_text


class SpeechOutputTests(unittest.TestCase):
    def _report(self):
        return {
            "prediction_date": "2026-08-15",
            "data_coverage": {
                "local_staleness_days": 20,
                "modern_team_count": 30,
                "modern_player_count": 4,
            },
            "top_win_picks": [{
                "team": "AAA", "opponent": "BBB", "probability": 0.61, "edge": 0.04,
                "metric_card": {"expected_value": 0.08, "market_price": 110},
            }],
            "top_batter_picks": [{
                "batter": "Test Hitter", "team": "AAA",
                "metric_card": {
                    "ai_probability": 0.59, "last10": 0.60,
                    "modern_context": {"available": False},
                },
            }],
            "games": [{
                "home_team": "AAA", "away_team": "BBB",
                "props": {
                    "totals": {
                        "pick": "over", "projected_total": 9.1,
                        "market_line": 8.5, "model_over_prob": 0.62,
                    },
                    "nrfi_prob": 0.5,
                },
            }],
        }

    def test_narration_is_detailed_and_mentions_uncertainty(self):
        text = build_narration(self._report(), style="sarcastic")
        self.assertIn("2026-08-15", text)
        self.assertIn("AAA over BBB", text)
        self.assertIn("baseball", text.lower())
        self.assertIn("old", text.lower())
        self.assertIn("NRFI", text)

    def test_print_backend_is_safe_for_dry_runs(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(speak_text("hello", backend="print"), "print")


if __name__ == "__main__":
    unittest.main()
