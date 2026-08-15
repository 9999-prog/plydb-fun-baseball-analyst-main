# MLB Baseball Analyst

A Statcast-backed MLB matchup and prop-analysis project. The project builds
plate-appearance features, rolls them into batter/game and team/game views,
trains hit and win models, and prints matchup/market diagnostics.

## Safe workflow

Run the audit before rebuilding models:

```powershell
python stat_audit.py
python stat_audit.py --json-out audit.json
python -m unittest discover -s tests -v
```

Build the advanced game-level feature table after the local PA parquet exists:

```powershell
python build_advanced_matchup_features.py
```

The audit labels metrics from an already-trained artifact as **in-sample**
and also fits a fresh copy of the estimator on prior seasons, evaluating only
the latest season. That time-held-out number is the more useful sanity check,
though it is still not a betting guarantee. Use `--no-temporal` only when you
need a fast structural audit.

## Current-season blending

`modern_stats.py` retrieves team hitting and pitching aggregates from the
public MLB Stats API and caches them for the process. Total-runs projections
use:

- **70% current-season / 30% historical** once the current season has at least
  10 games of support;
- a smaller modern weight for short current-season samples, so one hot/cold
  box score cannot dominate;
- historical-only or neutral fallback when the API/current-season data is
  unavailable.

The adapter uses runs per team game, not raw runs, so teams with different
numbers of games remain comparable. NRFI/RIFI remains neutral when the local
Statcast table has no recent first-inning sample; the team season endpoint does
not expose inning splits and the model does not invent them.

## Running the predictor and reading the cards

Run a dated slate explicitly when you want a reproducible report:

```powershell
python -u predict_todays_games.py 2026-08-15
```

The command writes `prediction_report.json` (ignored by Git) with the same
cards printed to the console. It includes the projected home/away score,
total and margin, raw model probability, confidence-adjusted probability,
no-vig market probability, probability edge, expected value per unit, H2H
sample status, data freshness, and source metadata. `PREDICTOR_DEBUG_TIMING=1`
can be set temporarily to print per-matchup timing; it is off by default.

The batter cards keep game-level hit rates separate from plate-appearance
rates. `L5`/`L10` means at least one hit in a completed game; `*_PA_hit_rate`
is the different PA-level statistic. `impact_rtg` uses real xBA/xwOBA and
hard-hit context when those fields exist, `min_ceiling` is a Wilson lower
bound, and `sponge_coeff` is intentionally `null` because that screenshot
label has no standard, validated definition. The modern player API provides
real current-season PA/hit context, but its game-hit conversion is explicitly
labelled a proxy rather than presented as an observed stat.

Market prices use median vig-adjusted probabilities. An obvious stale price
(outside a generous probability tolerance) is excluded from best-price EV;
the report exposes `market_price_filter` and `market_outlier_count`. This is
important because a single malformed `+2500` quote should not turn a normal
MLB matchup into a fake high-confidence edge.

### Optional spoken briefing

The normal predictor remains silent except for its text output. To hear a
sarcastic but evidence-aware briefing after the report is written:

```powershell
# Zero-install Windows fallback through SAPI:
python speak_predictions.py --backend sapi --style sarcastic --print-text

# Or let the module try pyttsx3 first, then Windows SAPI:
python speak_predictions.py --backend auto --style sarcastic

# Generate the report and speak it in one command:
python predict_todays_games.py 2026-08-15 --speak
```

`requirements-tts.txt` contains the optional offline `pyttsx3` dependency:
`python -m pip install -r requirements-tts.txt`. Windows SAPI remains the
fallback if that package is not installed. Use `--style straight` for a
plain briefing, `--voice` for a substring of an installed voice name, and
`--rate 160` or `--rate 200` to slow down or speed up delivery. Speech reads
only the local secret-free JSON report; it never reads `.env` or sends report
content to a remote voice service.

### Terminal display theme

The visible predictor and speech CLI output uses a two-colour theme: bright
white for headings, labels, explanations, and borders, and purple for
probabilities, prices, percentages, and other emphasized values. ANSI colours
are automatically disabled when output is redirected to a file. Set
`PREDICTOR_COLOR=0` or `NO_COLOR=1` to disable them manually.

On a compatible interactive Windows console, startup makes a best-effort
attempt to enlarge the current console font by 40% while preserving the
selected typeface. Modern Windows Terminal profiles may ignore the legacy
console API; if that happens, set the profile's font size manually. The
automatic attempt can be disabled with `PREDICTOR_FONT_SCALE=1.0` or
`PREDICTOR_RESIZE_FONT=0`.

## Why a single box-score stat cannot decide a game

A final box score is an outcome summary, not a clean pregame cause. A team can
have more hits and still lose because hits are sequenced differently, runners
are stranded, a home run occurs in a low-leverage inning, the bullpen gives up
the decisive run, or defense/baserunning changes the run value. A batter's hit
also does not determine the other eight lineup spots, opposing pitcher, park,
weather, defense, or bullpen context.

Raw counts also mix together opponent quality, park effects, schedule strength,
playing time, and luck. Small samples make this worse: a 4-for-10 stretch is a
useful observation but not a stable estimate of true talent. The model therefore
uses rolling rates, expected stats, pitcher context, platoon/park context,
shrinkage, and time-aware validation rather than one -boxed- number.

Never use final-game fields such as `events`, `is_hit`, `runs_on_pa`, or final
scores as pregame features. They are valid targets or historical observations,
but using the game being predicted is leakage.

## Leakage controls added

- Park factors are now shifted, season-to-date, pregame factors; the old
  season-wide factor used future games.
- Team lineup strength uses the first PA for each batter/game instead of
  averaging later same-game rows.
- The advanced feature builder uses first batter/game rows and starter-only
  rows for pregame features.
- `stat_audit.py` flags likely post-game feature names, missingness, duplicate
  keys, stale data, calibration, and constant features.

## Data freshness and security

Large raw and derived Statcast parquet files are intentionally kept local and
ignored by Git; this prevents a normal GitHub clone from carrying hundreds of
megabytes of generated data. The local workspace used for validation covers
**2023-03-30 through 2025-10-05**. Place a fresh Statcast export under
`data/pybaseball/statcast/` or rebuild it with the project’s data pipeline
before relying on player-level signals. The PA pipeline retains terminal score
fields and derives `runs_on_pa`, so recent run form and bullpen rates use real
score deltas instead of silently falling back to zero. The current-season API
adapter helps with team run-rate totals, but it cannot replace fresh
player-level Statcast or current first-inning splits. A stale local file should
reduce confidence, not produce identical confident picks.

A local `.env` file must not be committed. The repository previously contained
an API key in tracked configuration; rotate that key and remove the secret from
repository history, then use `.env.example` for future setup.

## Independent AI Reviewer handoff

`AI_REVIEW_PROMPT.md` is a copy-paste review request for an independent **AI
Reviewer**. Attach only source, tests, documentation, and a secret-free report;
never attach `.env` or credential-bearing logs. The prompt asks for a file and
line-level audit of leakage, calibration, target semantics, market math,
freshness, security, output quality, and regression tests. `CODE_REVIEW.md`
contains the same priorities as a maintained review checklist.