"""
Team Stats Integration Module

Integrates the mlb_game_predictor blended_stats.csv (71,575 games, 2010-2025)
as additional features for the plydb-fun-baseball-analyst-main project.

This provides team-level offensive/defensive metrics from FanGraphs:
- Offensive: avg, obp, slg, woba, wrc_plus, war, k_pct, bb_pct
- Defensive: k_per_9, bb_per_9, hr_per_9, era, fip, owar
- Blended with preseason projections based on games played
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

BLENDED_STATS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "blended_team_stats.csv"
)

# Column definitions from mlb_game_predictor
OFF_COLS = ["avg", "obp", "slg", "woba", "wrc_plus", "war", "k_pct", "bb_pct"]
DEF_COLS = ["k_per_9", "bb_per_9", "hr_per_9", "era", "fip", "owar"]
ALL_STAT_COLS = OFF_COLS + DEF_COLS

# Team abbreviation mapping (mlb_game_predictor uses full names, plydb uses abbreviations)
TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago White Sox": "CHW",
    "Chicago Cubs": "CHC",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYY",
    "New York Mets": "NYM",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

ABBR_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}


class TeamStatsIntegrator:
    """Loads and provides access to blended team stats for feature enrichment."""
    
    def __init__(self):
        self.stats_df = None
        self._load_stats()
    
    def _load_stats(self):
        """Load blended stats CSV."""
        try:
            if os.path.exists(BLENDED_STATS_PATH):
                self.stats_df = pd.read_csv(BLENDED_STATS_PATH)
                # Add abbreviation column
                self.stats_df['offensive_team_abbr'] = self.stats_df['offensive_team'].map(TEAM_NAME_TO_ABBR)
                self.stats_df['defensive_team_abbr'] = self.stats_df['defensive_team'].map(TEAM_NAME_TO_ABBR)
                print(f"Loaded {len(self.stats_df):,} blended team stats records")
                print(f"Date range: {self.stats_df['game_date'].min()} to {self.stats_df['game_date'].max()}")
            else:
                print(f"Blended stats file not found: {BLENDED_STATS_PATH}")
        except Exception as e:
            print(f"Error loading blended stats: {e}")
    
    def get_latest_team_stats(self, team_abbr, as_of_date=None):
        """
        Get the most recent blended stats for a team as of a given date.
        
        Args:
            team_abbr: Team abbreviation (e.g., 'LAD', 'NYY')
            as_of_date: Date string 'YYYY-MM-DD' or datetime, defaults to latest available
            
        Returns:
            Dict with offensive and defensive stats, or None if not found
        """
        if self.stats_df is None:
            return None
        
        # Find team in offensive role (most recent game)
        team_name = ABBR_TO_TEAM_NAME.get(team_abbr, team_abbr)
        
        team_games = self.stats_df[
            (self.stats_df['offensive_team'] == team_name) | 
            (self.stats_df['defensive_team'] == team_name)
        ].copy()
        
        if team_games.empty:
            return None
        
        # Filter by date if provided
        if as_of_date:
            if isinstance(as_of_date, str):
                as_of_date = pd.to_datetime(as_of_date)
            team_games = team_games[pd.to_datetime(team_games['game_date']) <= as_of_date]
        
        if team_games.empty:
            return None
        
        # Get most recent game
        latest = team_games.sort_values('game_date').iloc[-1]
        
        # Determine if team was offensive or defensive in that game
        is_offensive = latest['offensive_team'] == team_name
        
        if is_offensive:
            off_stats = {col: latest[col] for col in OFF_COLS if col in latest}
            # For defensive stats, we need the opponent's defensive stats from that game
            # But the CSV only has defensive stats from the opponent's perspective
            # So we'll use the defensive stats from when this team was defensive
            def_games = self.stats_df[self.stats_df['defensive_team'] == team_name]
            if not def_games.empty:
                def_latest = def_games.sort_values('game_date').iloc[-1]
                def_stats = {col: def_latest[col] for col in DEF_COLS if col in def_latest}
            else:
                def_stats = {}
        else:
            def_stats = {col: latest[col] for col in DEF_COLS if col in latest}
            off_games = self.stats_df[self.stats_df['offensive_team'] == team_name]
            if not off_games.empty:
                off_latest = off_games.sort_values('game_date').iloc[-1]
                off_stats = {col: off_latest[col] for col in OFF_COLS if col in off_latest}
            else:
                off_stats = {}
        
        return {
            'team': team_abbr,
            'as_of_date': latest['game_date'],
            'games_played': int(latest['games']),
            'offensive': off_stats,
            'defensive': def_stats,
            'win_flag': int(latest.get('win_flag', 0))
        }
    
    def get_matchup_stats(self, home_abbr, away_abbr, as_of_date=None):
        """
        Get blended stats for a specific matchup.
        
        Returns offensive stats for home vs away defensive, and vice versa.
        """
        home_stats = self.get_latest_team_stats(home_abbr, as_of_date)
        away_stats = self.get_latest_team_stats(away_abbr, as_of_date)
        
        if not home_stats or not away_stats:
            return None
        
        return {
            'home': home_stats,
            'away': away_stats,
            'matchup_date': as_of_date or datetime.now().strftime('%Y-%m-%d')
        }
    
    def get_team_season_averages(self, team_abbr, season_year):
        """Get season-averaged stats for a team in a specific year."""
        if self.stats_df is None:
            return None
        
        team_name = ABBR_TO_TEAM_NAME.get(team_abbr, team_abbr)
        
        season_data = self.stats_df[
            (self.stats_df['offensive_team'] == team_name) & 
            (pd.to_datetime(self.stats_df['game_date']).dt.year == season_year)
        ]
        
        if season_data.empty:
            return None
        
        # Average offensive stats
        off_avg = season_data[OFF_COLS].mean().to_dict()
        
        # Average defensive stats (when team was defensive)
        def_data = self.stats_df[
            (self.stats_df['defensive_team'] == team_name) & 
            (pd.to_datetime(self.stats_df['game_date']).dt.year == season_year)
        ]
        def_avg = def_data[DEF_COLS].mean().to_dict() if not def_data.empty else {}
        
        return {
            'team': team_abbr,
            'season': season_year,
            'games': len(season_data),
            'offensive_avg': off_avg,
            'defensive_avg': def_avg
        }


# Convenience function for quick access
def load_blended_stats():
    """Load the full blended stats DataFrame."""
    if os.path.exists(BLENDED_STATS_PATH):
        return pd.read_csv(BLENDED_STATS_PATH)
    return None


def get_team_strength_features(home_abbr, away_abbr, as_of_date=None):
    """
    Get team strength features for a matchup, formatted for ML model input.
    
    Returns dict with features compatible with win_model features:
    - home_batting_hit_rate, home_batting_xba (approximated from avg, obp)
    - away_batting_hit_rate, away_batting_xba
    - home_pitcher_* and away_pitcher_* (approximated from team defensive stats)
    """
    integrator = TeamStatsIntegrator()
    matchup = integrator.get_matchup_stats(home_abbr, away_abbr, as_of_date)
    
    if not matchup:
        return None
    
    home = matchup['home']
    away = matchup['away']
    
    # Map FanGraphs stats to win_model feature names
    # Note: These are approximations since win_model uses pitch-level rolling stats
    features = {}
    
    # Offensive features (batting strength)
    h_off = home['offensive']
    a_off = away['offensive']
    
    features['home_batting_hit_rate'] = h_off.get('avg', 0.250)
    features['home_batting_xba'] = h_off.get('woba', 0.320)  # wOBA as xBA proxy
    features['away_batting_hit_rate'] = a_off.get('avg', 0.250)
    features['away_batting_xba'] = a_off.get('woba', 0.320)
    
    # Defensive features (pitching strength) - use team defensive stats as pitcher proxy
    h_def = home['defensive']
    a_def = away['defensive']
    
    # Map team defensive stats to pitcher rolling stats
    features['home_pitcher_roll_k_rate'] = h_def.get('k_per_9', 8.5) / 9.0  # K/9 to rate
    features['home_pitcher_roll_bb_rate'] = h_def.get('bb_per_9', 3.0) / 9.0
    features['home_pitcher_roll_xba_allowed'] = 0.320 - (h_def.get('era', 4.0) - 4.0) * 0.01  # Rough approximation
    features['home_pitcher_roll_hardhit_allowed'] = h_def.get('hr_per_9', 1.2) / 9.0 * 0.15
    features['home_pitcher_roll_velo'] = 93.0  # Not available in team stats
    features['home_pitcher_roll_spin'] = 2300  # Not available
    
    features['away_pitcher_roll_k_rate'] = a_def.get('k_per_9', 8.5) / 9.0
    features['away_pitcher_roll_bb_rate'] = a_def.get('bb_per_9', 3.0) / 9.0
    features['away_pitcher_roll_xba_allowed'] = 0.320 - (a_def.get('era', 4.0) - 4.0) * 0.01
    features['away_pitcher_roll_hardhit_allowed'] = a_def.get('hr_per_9', 1.2) / 9.0 * 0.15
    features['away_pitcher_roll_velo'] = 93.0
    features['away_pitcher_roll_spin'] = 2300
    
    # Park factors (not in blended stats, use defaults)
    features['park_hit_factor'] = 1.0
    features['park_hr_factor'] = 1.0
    
    return features


if __name__ == "__main__":
    # Test the integration
    integrator = TeamStatsIntegrator()
    
    if integrator.stats_df is not None:
        print("\n--- Testing team stats lookup ---")
        stats = integrator.get_latest_team_stats('LAD')
        if stats:
            print(f"LAD latest stats (as of {stats['as_of_date']}):")
            print(f"  Games: {stats['games_played']}")
            print(f"  Offensive: avg={stats['offensive'].get('avg', 'N/A'):.3f}, wOBA={stats['offensive'].get('woba', 'N/A'):.3f}")
            print(f"  Defensive: ERA={stats['defensive'].get('era', 'N/A'):.2f}, K/9={stats['defensive'].get('k_per_9', 'N/A'):.1f}")
        
        print("\n--- Testing matchup stats ---")
        matchup = integrator.get_matchup_stats('LAD', 'NYY')
        if matchup:
            print(f"LAD vs NYY matchup stats available")
        
        print("\n--- Testing feature extraction ---")
        features = get_team_strength_features('LAD', 'NYY')
        if features:
            print(f"Generated {len(features)} features for win_model")
            for k, v in features.items():
                print(f"  {k}: {v:.4f}")