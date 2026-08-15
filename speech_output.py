"""Optional terminal narration and text-to-speech for prediction reports.

Speech is opt-in and reads only the secret-free prediction_report.json. It
never changes the statistical model or requires an audio backend for normal
prediction runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from terminal_theme import configure_terminal_display

DEFAULT_REPORT = Path(__file__).with_name("prediction_report.json")


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pct(value: Any, digits: int = 1) -> str:
    number = _number(value)
    return "unknown" if number is None else f"{number * 100:.{digits}f} percent"


def _read_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("prediction report must contain a JSON object")
    return report


def _card_for_pick(pick: Mapping[str, Any]) -> Mapping[str, Any]:
    card = pick.get("metric_card")
    return card if isinstance(card, Mapping) else {}


def build_narration(
    report: Mapping[str, Any], *, style: str = "sarcastic", max_games: int = 4
) -> str:
    """Turn report evidence into a detailed, funny-but-direct briefing."""
    sarcastic = str(style or "straight").lower() in {"sarcastic", "funny", "dry"}
    date = report.get("prediction_date", "the requested slate")
    coverage = report.get("data_coverage", {})
    if not isinstance(coverage, Mapping):
        coverage = {}
    lines = [f"MLB briefing for {date}."]
    if sarcastic:
        lines.append(
            "Straight answer first: this is a probability model, not a crystal ball "
            "with a tiny baseball cap."
        )
    stale = _number(coverage.get("local_staleness_days"))
    modern_teams = coverage.get("modern_team_count", 0)
    modern_players = coverage.get("modern_player_count", 0)
    if stale is not None and stale > 14:
        lines.append(
            f"The local Statcast player file is {stale:.0f} days old. Current-season "
            "team context is live, but old player form is background evidence, not breaking news."
        )
    else:
        lines.append(
            f"Live context covers {modern_teams} teams and {modern_players} player records where available."
        )

    wins = report.get("top_win_picks", [])
    if isinstance(wins, list) and wins:
        lines.append("Top win angles:")
        for pick in wins[:2]:
            if not isinstance(pick, Mapping):
                continue
            card = _card_for_pick(pick)
            price = _number(card.get("market_price"))
            price_text = f" at market price {int(price):+d}" if price is not None else " with no usable market price"
            lines.append(
                f"{pick.get('team', 'Unknown team')} over {pick.get('opponent', 'unknown opponent')}: "
                f"raw win probability {_pct(pick.get('probability'))}, market edge {_pct(pick.get('edge'))}, "
                f"expected value {_pct(card.get('expected_value'))}{price_text}."
            )
    else:
        lines.append(
            "There are no win picks clearing the evidence threshold. The model chose restraint, "
            "which is less glamorous and usually healthier."
        )

    batters = report.get("top_batter_picks", [])
    if isinstance(batters, list) and batters:
        lines.append("Top batter hit angles, with the important reality check:")
        for pick in batters[:2]:
            if not isinstance(pick, Mapping):
                continue
            card = _card_for_pick(pick)
            modern = card.get("modern_context", {})
            modern_text = (
                f"current-season PA hit rate {_pct(modern.get('pa_hit_rate'))}"
                if isinstance(modern, Mapping) and modern.get("available")
                else "no current-season player context"
            )
            lines.append(
                f"{pick.get('batter', 'Unknown hitter')} of {pick.get('team', 'unknown team')}: "
                f"confidence-adjusted hit probability {_pct(card.get('ai_probability'))}, "
                f"last-ten game hit rate {_pct(card.get('last10'))}, {modern_text}."
            )
        if sarcastic:
            lines.append(
                "A ten-game hit streak is still ten games, not a promotion to the Hall of Fame. "
                "Baseball remains aggressively rude that way."
            )
    else:
        lines.append("No batter hit spot cleared the evidence filter today.")

    total_angles = []
    neutral_first_inning = 0
    games = report.get("games", [])
    if isinstance(games, list):
        for game in games:
            if not isinstance(game, Mapping):
                continue
            props = game.get("props", {})
            if not isinstance(props, Mapping):
                continue
            totals = props.get("totals", {})
            if isinstance(totals, Mapping) and str(totals.get("pick", "PASS")).upper() != "PASS":
                total_angles.append((game, totals))
            nrfi = _number(props.get("nrfi_prob"))
            if nrfi is not None and abs(nrfi - 0.5) < 0.03:
                neutral_first_inning += 1
    if total_angles:
        lines.append("Totals with a non-neutral model direction:")
        for game, totals in total_angles[:max(1, int(max_games))]:
            projected = _number(totals.get("projected_total"))
            line = _number(totals.get("market_line"))
            lines.append(
                f"{game.get('away_team', 'away')} at {game.get('home_team', 'home')}: "
                f"{str(totals.get('pick', 'PASS')).upper()}, projected total "
                f"{projected:.1f} against market line {line:.1f}, model over probability "
                f"{_pct(totals.get('model_over_prob'))}."
                if projected is not None and line is not None
                else "with incomplete total-market data."
            )
    if neutral_first_inning:
        lines.append(
            f"First-inning note: {neutral_first_inning} matchup(s) are effectively neutral for NRFI versus RIFI "
            "because the local first-inning sample does not support a stronger claim. No, the model will not "
            "make up a starter narrative just to sound confident."
        )
    lines.append(
        "Final verdict: use probability, evidence quality, and price together; pass when the data is stale or "
        "the edge is too small. Baseball has a very deep bullpen."
        if sarcastic
        else "Final verdict: compare probability, evidence quality, and price; pass when the edge is too small."
    )
    return " ".join(str(line).strip() for line in lines if str(line).strip())


def _speak_pyttsx3(text: str, *, voice: str | None = None, rate: int = 180) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", int(rate))
    if voice:
        wanted = voice.lower()
        for candidate in engine.getProperty("voices") or []:
            identity = " ".join(str(getattr(candidate, field, "")) for field in ("id", "name")).lower()
            if wanted in identity:
                engine.setProperty("voice", candidate.id)
                break
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def _speak_sapi(text: str, *, voice: str | None = None, rate: int = 180) -> None:
    if platform.system().lower() != "windows":
        raise RuntimeError("Windows SAPI is only available on Windows")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise RuntimeError("PowerShell was not found")
    script = r"""$ErrorActionPreference = "Stop";
Add-Type -AssemblyName System.Speech;
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$requested = $env:MLB_TTS_VOICE;
if ($requested) {
  $voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Name -like ("*" + $requested + "*") } | Select-Object -First 1;
  if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name); }
}
$mappedRate = [Math]::Max(-10, [Math]::Min(10, [int](($env:MLB_TTS_RATE - 180) / 15)));
$synth.Rate = $mappedRate;
$synth.Speak([Console]::In.ReadToEnd());
$synth.Dispose();"""
    env = os.environ.copy()
    if voice:
        env["MLB_TTS_VOICE"] = str(voice)
    env["MLB_TTS_RATE"] = str(int(rate))
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
        input=text, text=True, capture_output=True, timeout=300, env=env, check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "SAPI speech failed").strip()
        raise RuntimeError(detail)


def speak_text(text: str, *, backend: str = "auto", voice: str | None = None, rate: int = 180) -> str:
    """Speak text and return the backend used."""
    backend = str(backend or "auto").lower()
    if backend == "print":
        print(text)
        return "print"
    attempts = ["pyttsx3", "sapi"] if backend == "auto" else [backend]
    errors = []
    for candidate in attempts:
        try:
            if candidate == "pyttsx3":
                _speak_pyttsx3(text, voice=voice, rate=rate)
            elif candidate == "sapi":
                _speak_sapi(text, voice=voice, rate=rate)
            else:
                raise ValueError(f"unknown speech backend: {candidate}")
            return candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("No speech backend succeeded. " + " | ".join(errors))


def speak_report_file(
    report_path: str | os.PathLike[str] = DEFAULT_REPORT,
    *, style: str = "sarcastic", backend: str = "auto", voice: str | None = None,
    rate: int = 180, max_games: int = 4, print_text: bool = False,
) -> str:
    report = _read_report(report_path)
    text = build_narration(report, style=style, max_games=max_games)
    if print_text:
        print("\n--- spoken briefing ---\n" + text + "\n--- end briefing ---")
    return speak_text(text, backend=backend, voice=voice, rate=rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speak an MLB prediction report.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--style", choices=["straight", "sarcastic"], default="sarcastic")
    parser.add_argument("--backend", choices=["auto", "pyttsx3", "sapi", "print"], default="auto")
    parser.add_argument("--voice", default=None, help="Optional substring of an installed voice name.")
    parser.add_argument("--rate", type=int, default=180, help="Approximate words per minute.")
    parser.add_argument("--max-games", type=int, default=4)
    parser.add_argument("--print-text", action="store_true", help="Print narration before speaking.")
    args = parser.parse_args(argv)
    configure_terminal_display()
    try:
        backend = speak_report_file(
            args.report, style=args.style, backend=args.backend, voice=args.voice,
            rate=args.rate, max_games=args.max_games, print_text=args.print_text,
        )
        print(f"Speech briefing complete using backend: {backend}")
        return 0
    except Exception as exc:
        print(f"Speech briefing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
