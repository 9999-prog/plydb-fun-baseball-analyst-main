# MLB Analyst - prioritized review for the AI Reviewer

This is a review map, not a claim that the model is calibrated or profitable.
Do not retrain until the temporal holdout, leakage audit, and calibration work
below are complete.

## P0 - correctness and trust

1. **Secret handling and reproducibility**
   - Keep credentials only in a local `.env`; never print request exceptions or
     commit the file. Rotate any key that has appeared in chat, logs, or Git
     history.
   - Treat a missing or rejected market feed as `NO_MARKET`, never as zero edge
     or positive expected value.

2. **Temporal leakage audit**
   - Review `build_advanced_matchup_features.py`, `build_prediction_caches.py`,
     `build_win_model.py`, and `save_model_and_snapshots.py` for every rolling
     feature and join. Every feature for game date `t` must use rows strictly
     before `t` and must have an explicit `as_of_date`.
   - Check park factors, starter identification, bullpen rows, lineup rows,
     and target extraction by `game_pk`. Add duplicate-key and future-join
     assertions.

3. **Probability calibration**
   - The hand-weighted logit blend in `predict_todays_games.py` is a scoring
     rule, not a fitted probability model. Evaluate chronological folds with
     log loss, Brier score, calibration slope/intercept, reliability bins, and
     prediction intervals. Compare calibration methods only on validation data.
   - Replace any neutral or missing-feature fallback with an explicit status;
     never use a home-favoring fallback such as `0.54`.

4. **Player-hit target semantics**
   - Keep PA hit rate separate from game-level probability of at least one hit.
     Model expected PA, lineup position, confirmed lineup status, and pitcher
     availability. A perfect recent rate or tiny BvP sample must not create a
     high-confidence pick.

## P1 - model quality

- Rebuild `build_win_model.py` with date-based cross-validation rather than one
  fixed train/test split. Persist the feature schema, training cutoff,
  calibration metrics, and class balance with the artifact.
- Validate `save_model_and_snapshots.py` freshness and `as_of_date`; snapshots
  must be reproducible for any historical forecast date.
- Refactor `build_prediction_caches.py` into functions under a `main()` guard;
  remove broad warning suppression, repeated transformations, and silent
  missing-column behavior.
- In `modern_stats.py`, add retries/backoff, HTTP status handling, source
  timestamps, cache TTLs, and clear separation between current-season team
  data and current-season player data. Do not turn API failures into fabricated
  player estimates.
- In `prop_metrics.py`, validate score-rate denominators, team/game counts,
  market points, vig normalization, and over/under direction. Keep NRFI/RIFI
  neutral without enough first-inning observations.
- Recheck bullpen quality: runs per PA is hard to interpret; consider runs per
  nine, FIP/xFIP/SIERA-like inputs, leverage, rest, and reliever availability.
- Confirm H2H/BvP effective sample sizes and shrinkage priors. Historical matchup
  data is a weak prior, not a standalone causal signal.

## P1 - engineering quality

- Add schema validation at every parquet/joblib boundary. `pandera` or
  `pydantic` can express required columns and bounds; fail closed on bad data.
- Add `pytest`/`hypothesis` tests for date boundaries, missing fields, malformed
  odds, aliases, stale data, and no-market behavior. Run `ruff` and `pyright`.
- Separate network adapters from scoring and presentation. The predictor
  remains mostly top-level, which makes unit testing and repeated forecasts
  expensive.
- Use a cache with TTL and an atomic write/rename for reports. Include source
  timestamps, request date, and a run identifier in JSON.

## P2 - statistical extensions

- Add confirmed lineups, batting-order slot, expected PA, handedness-specific
  xwOBA, barrel/contact quality, pitch-mix matchup, park/weather, catcher,
  defense, baserunning, and bullpen availability only when time-safe data is
  available.
- Consider a hierarchical Bayesian model for player talent and partial pooling.
  Use an interpretable baseline and keep explanation tools diagnostic, not causal.
- Use a faster dataframe library only if profiling shows pandas is the
  bottleneck. Avoid adding dependencies for appearance alone.

## AI Reviewer instructions

Ask the AI Reviewer to act as a hostile but constructive senior MLB statistician
and production ML reviewer. Give it the repository tree, the current report if
present, and test output. Tell it to read code, not infer behavior from
filenames.

Require these sections:

1. **Executive verdict**: what is safe to trust, what is not, and whether the
   project should make picks with the current data freshness.
2. **P0/P1/P2 findings**: file, function, line range or exact excerpt, failure
   mode, why it matters, minimal fix, and regression test.
3. **Leakage proof**: trace each feature from raw data to prediction and show
   that its maximum timestamp is strictly before the forecast game. Identify
   future joins, post-game fields, same-game lineup rows, park leakage, and
   target contamination.
4. **Calibration audit**: specify chronological splits, metrics, reliability
   bins, confidence intervals, and a go/no-go threshold for displaying 60%,
   70%, or 80% probabilities. Separate discrimination from calibration.
5. **Player-hit audit**: verify PA hit rate, game-hit probability, expected PA,
   lineup slot, current-season data, BvP, platoon, and pitcher quality are not
   mixed as if they were the same target.
6. **Market audit**: check American odds parsing, vig removal, best-price
   selection, stale/outlier filtering, missing-market behavior, and EV math.
   Prove that an unavailable market cannot create an edge.
7. **Math review**: recompute Pythagorean expectation, run projections,
   bullpen rates, recency weighting, shrinkage, park multipliers, NRFI/RIFI,
   and score-to-win conversion with small hand-built examples.
8. **Data freshness**: list every source, timestamp, cache TTL, fallback, and
   whether the fallback is neutral, historical, current-season, or synthetic.
   Reject synthetic values unless clearly marked and justified.
9. **Output contract**: check that labels say `MODEL LEAN`, `VALUE EDGE`, or
   `PASS` appropriately; no edge label is allowed with `market N/A`; no stale
   player file may produce a high-confidence player pick; no secrets or raw
   credential-bearing URLs appear in logs; and there is no duplicate output or
   mojibake.
10. **Patch plan**: rank the smallest safe changes, add tests first where
    possible, and list any change that must wait for fresh data or retraining.

Challenge these assumptions instead of accepting them:
- 70% modern and 30% historical is a policy choice until validated;
- hand-set weights are not calibrated probabilities;
- H2H and BvP are weak, confounded samples;
- a box-score stat is an outcome summary, not a causal pregame signal;
- a model probability without a price is not a betting edge;
- a current-season team endpoint does not make stale player Statcast current;
- a neutral fallback is missing evidence, not a 50/50 observation.
