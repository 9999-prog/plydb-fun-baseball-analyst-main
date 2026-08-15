import os
import unittest
from unittest.mock import patch

import terminal_theme


class TerminalThemeTests(unittest.TestCase):
    def tearDown(self):
        terminal_theme.set_color_enabled(False)

    def test_enabled_theme_only_uses_white_and_purple(self):
        terminal_theme.set_color_enabled(True)

        white_text = terminal_theme.white("label")
        purple_text = terminal_theme.purple("52.4%")
        dim_text = terminal_theme.dim("secondary detail")

        self.assertIn(terminal_theme.ANSI_WHITE, white_text)
        self.assertIn(terminal_theme.ANSI_PURPLE, purple_text)
        self.assertIn("97", dim_text)
        self.assertNotIn("31m", white_text)  # no red
        self.assertNotIn("32m", purple_text)  # no green
        self.assertNotIn("33m", dim_text)  # no yellow

    def test_disabled_theme_returns_plain_text(self):
        terminal_theme.set_color_enabled(False)

        self.assertEqual(terminal_theme.white("label"), "label")
        self.assertEqual(terminal_theme.bold("label"), "label")
        self.assertEqual(terminal_theme.purple("52.4%"), "52.4%")

    def test_font_scale_defaults_to_forty_percent_increase(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PREDICTOR_FONT_SCALE", None)
            self.assertAlmostEqual(
                terminal_theme._requested_font_scale(), 1.40
            )

    def test_font_scale_can_be_disabled(self):
        with patch.dict(os.environ, {"PREDICTOR_FONT_SCALE": "1.0"}):
            self.assertAlmostEqual(
                terminal_theme._requested_font_scale(), 1.0
            )
        with patch.dict(os.environ, {"PREDICTOR_FONT_SCALE": "0"}):
            self.assertIsNone(terminal_theme._requested_font_scale())


if __name__ == "__main__":
    unittest.main()
