from __future__ import annotations

import unittest
from pathlib import Path

import requests

from predictor_utils import clean_sentence, pick_annotation, safe_error


class PredictorUtilityTests(unittest.TestCase):
    def test_safe_error_redacts_key_and_query_secret(self):
        key = "secret-test-value"
        error = requests.RequestException(
            "401 Client Error: Unauthorized for https://example.test/odds?apiKey="
            + key
            + "&regions=au"
        )
        message = safe_error(error, secrets=[key])
        self.assertNotIn(key, message)
        self.assertIn("[REDACTED]", message)

    def test_http_auth_errors_are_generic(self):
        response = requests.Response()
        response.status_code = 401
        error = requests.HTTPError("bad request URL with apiKey=secret", response=response)
        self.assertEqual(safe_error(error, secrets=["secret"]), "HTTP 401 credentials rejected")

    def test_query_redaction_works_without_explicit_secret_argument(self):
        error = requests.RequestException(
            "request failed for https://example.test/odds?apiKey=secret-test-value"
        )
        message = safe_error(error)
        self.assertNotIn("secret-test-value", message)
        self.assertIn("apiKey=[REDACTED]", message)

    def test_predictor_api_queries_use_request_params(self):
        source = (Path(__file__).resolve().parents[1] / "predict_todays_games.py").read_text(
            encoding="utf-8"
        )
        for malformed in ('f"-stats=', 'f"-sportId=', "roster-season="):
            self.assertNotIn(malformed, source)
        self.assertIn('"hydrate": "probablePitcher,team"', source)

    def test_missing_market_never_gets_edge_label(self):
        label = pick_annotation(0.70, edge=None, signal_quality=0.9, market_available=False)
        self.assertEqual(label, "STRONG MODEL LEAN")
        self.assertNotIn("EDGE", label)

    def test_weak_evidence_passes(self):
        self.assertEqual(pick_annotation(0.80, signal_quality=0.2), "PASS - WEAK DATA")

    def test_sentence_cleanup_removes_doubled_periods(self):
        self.assertEqual(clean_sentence("The bullpen helps.."), "The bullpen helps.")


if __name__ == "__main__":
    unittest.main()
