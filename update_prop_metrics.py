with open("prop_metrics.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace fair_american_odds with probability_to_decimal
old_fair = '''def fair_american_odds(probability):
    p = safe_num(probability)
    p = clamp(p, 1e-6, 0.999999)
    if p >= 0.5:
        return int(round((p / (1.0 - p)) * 100.0))
    return int(round(-((1.0 - p) / p) * 100.0))


def _american_implied_probability(odds):
    odds = safe_num(odds)
    if not valid(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)'''

new_fair = '''def probability_to_decimal(probability):
    """Convert probability to fair decimal odds."""
    p = safe_num(probability)
    p = clamp(p, 1e-6, 0.999999)
    return 1.0 / p


def decimal_to_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    odds = safe_num(decimal_odds)
    if not valid(odds) or odds <= 1.0:
        return np.nan
    return 1.0 / odds


# Backward compatibility (deprecated)
def fair_american_odds(probability):
    """DEPRECATED: Convert probability to fair American odds."""
    dec = probability_to_decimal(probability)
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    else:
        return int(round(-100.0 / (dec - 1.0)))


def _american_implied_probability(odds):
    """DEPRECATED: Convert American odds to implied probability."""
    odds = safe_num(odds)
    if not valid(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)'''

content = content.replace(old_fair, new_fair)

# Also update best_total_market to work with decimal odds
old_best_total = '''def best_total_market(home_team, away_team, odds_data, side="over"):
    """Return a robust best quote for one side of the main totals market.

    Totals are vulnerable to mixing alternate lines and stale bookmaker
    quotes. Prefer the consensus point and discard a quote whose implied
    probability is more than ten percentage points from the side's median.
    """
    candidates = []
    target = str(side).lower()
    for game in odds_data or []:
        if not ((team_name_matches(game.get("home_team"), home_team)
                 and team_name_matches(game.get("away_team"), away_team))
                or (team_name_matches(game.get("home_team"), away_team)
                    and team_name_matches(game.get("away_team"), home_team))):
            continue
        for bookmaker in game.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    if str(outcome.get("name", "")).lower() != target:
                        continue
                    point = safe_num(outcome.get("point"))
                    price = safe_num(outcome.get("price"))
                    implied = _american_implied_probability(price)
                    if not valid(point) or not valid(price) or not valid(implied):
                        continue
                    candidates.append({
                        "point": float(point),
                        "price": float(price),
                        "implied_probability": float(implied),
                    })
    if not candidates:
        return None

    consensus_point = float(np.median([row["point"] for row in candidates]))
    same_line = [
        row for row in candidates
        if abs(row["point"] - consensus_point) <= 0.01
    ]
    if not same_line:
        same_line = [min(
            candidates,
            key=lambda row: abs(row["point"] - consensus_point),
        )]

    consensus_probability = float(np.median([
        row["implied_probability"] for row in same_line
    ]))
    robust = [
        row for row in same_line
        if abs(row["implied_probability"] - consensus_probability) <= 0.10
    ]
    if not robust:
        robust = [min(
            same_line,
            key=lambda row: abs(row["implied_probability"] - consensus_probability),
        )]
    # At the same line, the largest American number is the best payout.
    return max(robust, key=lambda row: row["price"])'''

new_best_total = '''def best_total_market(home_team, away_team, odds_data, side="over"):
    """Return a robust best quote for one side of the main totals market (decimal odds).

    Totals are vulnerable to mixing alternate lines and stale bookmaker
    quotes. Prefer the consensus point and discard a quote whose implied
    probability is more than ten percentage points from the side's median.
    """
    candidates = []
    target = str(side).lower()
    for game in odds_data or []:
        if not ((team_name_matches(game.get("home_team"), home_team)
                 and team_name_matches(game.get("away_team"), away_team))
                or (team_name_matches(game.get("home_team"), away_team)
                    and team_name_matches(game.get("away_team"), home_team))):
            continue
        for bookmaker in game.get("bookmakers", []) or []:
            for market in bookmaker.get("markets", []) or []:
                if market.get("key") != "totals":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    if str(outcome.get("name", "")).lower() != target:
                        continue
                    point = safe_num(outcome.get("point"))
                    price = safe_num(outcome.get("price"))
                    implied = decimal_to_probability(price)
                    if not valid(point) or not valid(price) or not valid(implied):
                        continue
                    candidates.append({
                        "point": float(point),
                        "price": float(price),  # decimal odds
                        "implied_probability": float(implied),
                    })
    if not candidates:
        return None

    consensus_point = float(np.median([row["point"] for row in candidates]))
    same_line = [
        row for row in candidates
        if abs(row["point"] - consensus_point) <= 0.01
    ]
    if not same_line:
        same_line = [min(
            candidates,
            key=lambda row: abs(row["point"] - consensus_point),
        )]

    consensus_probability = float(np.median([
        row["implied_probability"] for row in same_line
    ]))
    robust = [
        row for row in same_line
        if abs(row["implied_probability"] - consensus_probability) <= 0.10
    ]
    if not robust:
        robust = [min(
            same_line,
            key=lambda row: abs(row["implied_probability"] - consensus_probability),
        )]
    # At the same line, the largest decimal odds is the best payout.
    return max(robust, key=lambda row: row["price"])'''

content = content.replace(old_best_total, new_best_total)

# Update the fair price calls in totals_pick and nrfi
old_totals_fair = '''    fair_over = fair_american_odds(totals["model_over_prob"])
    fair_under = fair_american_odds(totals["model_under_prob"])'''

new_totals_fair = '''    fair_over = probability_to_decimal(totals["model_over_prob"])
    fair_under = probability_to_decimal(totals["model_under_prob"])'''

content = content.replace(old_totals_fair, new_totals_fair)

old_nrfi_fair = '''    return {"under": under, "nrfi": nrfi}


def select_best_under(games, pa_df, as_of_date, odds_data, modern_stats=None):
    candidate = None
    for game in games:
        under = totals_pick(
            game["home_team"], game["away_team"], pa_df, as_of_date, odds_data,
            modern_stats=modern_stats,
        )
        if candidate is None or under["model_under_prob"] > candidate["model_under_prob"]:
            candidate = under
    return candidate


def select_best_nrfi(games, pa_df, as_of_date):
    candidate = None
    for game in games:
        prob = projected_nrfi_prob(game["home_team"], game["away_team"], pa_df, as_of_date)
        if candidate is None or prob > candidate["prob"]:
            candidate = {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "prob": prob,
                "fair_price": fair_american_odds(prob),
            }
    return candidate'''

new_nrfi_fair = '''    return {"under": under, "nrfi": nrfi}


def select_best_under(games, pa_df, as_of_date, odds_data, modern_stats=None):
    candidate = None
    for game in games:
        under = totals_pick(
            game["home_team"], game["away_team"], pa_df, as_of_date, odds_data,
            modern_stats=modern_stats,
        )
        if candidate is None or under["model_under_prob"] > candidate["model_under_prob"]:
            candidate = under
    return candidate


def select_best_nrfi(games, pa_df, as_of_date):
    candidate = None
    for game in games:
        prob = projected_nrfi_prob(game["home_team"], game["away_team"], pa_df, as_of_date)
        if candidate is None or prob > candidate["prob"]:
            candidate = {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "prob": prob,
                "fair_price": probability_to_decimal(prob),
            }
    return candidate'''

content = content.replace(old_nrfi_fair, new_nrfi_fair)

with open("prop_metrics.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done!")