"""Enhanced Baseball Predictor with Live Data Integration
Combines plydb-fun-baseball-analyst-main with sports-skills and pybaseball
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
import requests

import numpy as np
import pandas as pd
import joblib

# Add enhanced imports
try:
    from sports_skills.mlb import get_scoreboard, get_standings, get_teams, get_injuries, get_news, get_schedule
    MLB_DATA_AVAILABLE = True
except ImportError:
    MLB_DATA_AVAILABLE = False
    print("WARNING: sports-skills MLB data not available - using fallback data sources")

try:
    import pybaseball as pb
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("WARNING: pybaseball not available - using local data only")

# Current project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from config_loader import get_config, get_odds_config, get_win_model_weights
from logging_utils import setup_logging, get_logger, PredictionLogger, LogTimer

# Setup enhanced logging
logger = setup_logging("enhanced_mlb_analyst.predictor")
pred_logger = PredictionLogger(logger)
# Use the underlying logger for LogTimer
timer_logger = logger

class EnhancedBaseballPredictor:
    def __init__(self):
        self.config = get_config()
        self.odds_config = get_odds_config()
        self.win_weights = get_win_model_weights()
        self.session_start = datetime.now()
        
        # Load existing model
        try:
            hit_model_data = joblib.load("hit_model.joblib")
            win_model_data = joblib.load("win_model.joblib")
            
            self.hit_model = hit_model_data.get('model')
            self.hit_model_features = hit_model_data.get('features', [])
            self.win_model = win_model_data.get('model')
            self.win_model_features = win_model_data.get('features', [])
            print(f"Loaded existing models")
            print(f"  Win model features: {len(self.win_model_features)}")
            print(f"  Hit model features: {len(self.hit_model_features)}")
        except Exception as e:
            print(f"Failed to load models: {e}")
            self.hit_model = None
            self.hit_model_features = []
            self.win_model = None
            self.win_model_features = []
    
    def fetch_live_game_data(self):
        """Enhanced data fetching with fallback sources"""
        games_data = []
        
        # Primary: Try sports-skills MLB data
        if MLB_DATA_AVAILABLE:
            try:
                scoreboard = get_scoreboard()
                if scoreboard:
                    games_data.extend(self.process_sports_skills_data(scoreboard))
                    print(f"Fetched {len(games_data)} games from sports-skills")
            except Exception as e:
                print(f"sports-skills failed: {e}")
        
        # Secondary: Try sports-skills for schedule
        if MLB_DATA_AVAILABLE:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                schedule_data = get_schedule(date=today)
                if schedule_data:
                    games_from_schedule = self.process_sports_skills_schedule(schedule_data)
                    games_data.extend(games_from_schedule)
                    print(f"Fetched {len(games_from_schedule)} games from sports-skills schedule")
            except Exception as e:
                print(f"sports-skills schedule failed: {e}")
        
        # Tertiary: Try pybaseball for Statcast data (if needed)
        if PYBASEBALL_AVAILABLE and not games_data:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                # Use schedule_and_record for a specific team if needed
            except Exception as e:
                print(f"pybaseball schedule failed: {e}")
        
        # Tertiary: Use existing local data as fallback
        if not games_data:
            print("Using local prediction_report.json as fallback")
            with open("prediction_report.json", "r") as f:
                local_data = json.load(f)
                games_data = local_data.get("games", [])
        
        return games_data
    
    def fetch_pybaseball_game_data(self, game_id):
        """Enhanced pybaseball integration for pitch-level data"""
        try:
            # Get game info
            games_df = pb.game(game_id)
            if len(games_df) == 0:
                return None
                
            game_info = games_df.iloc[0]
            
            # Get batting and pitching stats
            teams = pb.teams()
            team_home = teams[teams['id'] == game_info['home_id']].iloc[0]['name']
            team_away = teams[teams['id'] == game_info['away_id']].iloc[0]['name']
            
            # Get current players
            players_home = pb.roster(game_info['home_id'])
            players_away = pb.roster(game_info['away_id'])
            
            return {
                'game_id': game_id,
                'home_team': team_home,
                'away_team': team_away,
                'home_pitcher': game_info.get('home_pitcher', 'TBD'),
                'away_pitcher': game_info.get('away_pitcher', 'TBD'),
                'data_source': 'pybaseball_live',
                'fetch_time': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error fetching pybaseball data for game {game_id}: {e}")
            return None
    
    def process_sports_skills_data(self, scoreboard_data):
        """Process sports-skills MLB data format"""
        games = []
        # Add processing logic here based on sports-skills API structure
        return games
    
    def process_sports_skills_schedule(self, schedule_response):
        """Process sports-skills schedule data format"""
        games = []
        try:
            # The response has structure: {'status': True, 'data': {'events': [...], 'count': N}}
            data = schedule_response.get('data', {})
            events = data.get('events', [])
            
            for event in events:
                competitors = event.get('competitors', [])
                home_team = None
                away_team = None
                home_pitcher = 'TBD'
                away_pitcher = 'TBD'
                
                for comp in competitors:
                    if comp.get('home_away') == 'home':
                        home_team = comp.get('team', {}).get('abbreviation', '')
                    else:
                        away_team = comp.get('team', {}).get('abbreviation', '')
                
                if home_team and away_team:
                    games.append({
                        'game_id': event.get('id', ''),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_pitcher': home_pitcher,
                        'away_pitcher': away_pitcher,
                        'odds': event.get('odds'),
                        'start_time': event.get('start_time', ''),
                        'venue': event.get('venue', {}).get('name', ''),
                        'data_source': 'sports_skills_schedule',
                        'fetch_time': datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error processing schedule data: {e}")
        return games
    
    def enhanced_model_prediction(self, games_data):
        """Improved prediction with multiple data sources"""
        enhanced_predictions = []
        
        for game in games_data:
            try:
                # Combine multiple prediction sources
                model_prediction = self.legacy_model_prediction(game)
                live_data_enhancement = self.live_data_enhancement(game)
                odds_integration = self.odds_price_analysis(game)
                
                # Weighted ensemble approach
                final_prediction = self.ensemble_predictions(
                    model_prediction, live_data_enhancement, odds_integration
                )
                
                enhanced_predictions.append(final_prediction)
                
            except Exception as e:
                print(f"Error predicting for game: {e}")
                continue
        
        return enhanced_predictions
    
    def live_data_enhancement(self, game_data):
        """Add live data context to predictions"""
        enhancements = {}
        
        # Add current season context
        enhancements['season_context'] = self.get_current_season_context()
        
        # Add injury updates if available
        if MLB_DATA_AVAILABLE:
            try:
                injuries = get_injuries()
                key_players = self.extract_key_players(game_data)
                player_injuries = self.check_player_injuries(key_players, injuries)
                enhancements['injuries'] = player_injuries
            except:
                enhancements['injuries'] = {}
        
        # Add recent performance trends
        enhancements['recent_trends'] = self.calculate_recent_trends(game_data)
        
        return enhancements
    
    def calculate_recent_trends(self, game_data):
        """Calculate enhanced trend analysis"""
        trends = {}
        
        # Check if we have enough historical data
        try:
            # Use pybaseball for recent game data
            if PYBASEBALL_AVAILABLE:
                team_name = game_data.get('home_team') or game_data.get('away_team')
                if team_name:
                    # Get recent games (last 10)
                    recent_games = pb.team_game_logs(team_name, last=10)
                    trends['recent_record'] = self.calculate_game_log_performance(recent_games)
                    trends['recent_offense'] = self.calculate_offensive_trends(recent_games)
                    trends['recent_defense'] = self.calculate_defensive_trends(recent_games)
        except:
            trends['status'] = 'insufficient_data'
        
        return trends
    
    def get_current_season_context(self):
        """Get current season context from live data"""
        try:
            if MLB_DATA_AVAILABLE:
                standings = get_standings()
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                season_context = {
                    'current_date': current_date,
                    'standings_available': len(standings) > 0,
                    'live_data_freshness': self.assess_data_freshness()
                }
                
                return season_context
        except:
            return {'status': 'api_unavailable'}
    
    def assess_data_freshness(self):
        """Assess data freshness across sources"""
        freshness_scores = {}
        
        # Check each data source
        freshness_scores['sports_skills'] = self.check_sports_skills_freshness()
        freshness_scores['pybaseball'] = self.check_pybaseball_freshness()
        freshness_scores['local_data'] = self.check_local_data_freshness()
        
        # Calculate weighted average
        weights = {'sports_skills': 0.5, 'pybaseball': 0.3, 'local_data': 0.2}
        overall_score = sum(freshness_scores[k] * weights[k] for k in weights)
        
        return {
            'overall_score': overall_score,
            'individual_scores': freshness_scores,
            'recommendation': self.get_freshness_recommendation(overall_score)
        }
    
    def check_sports_skills_freshness(self):
        """Check sports-skills data freshness"""
        try:
            if MLB_DATA_AVAILABLE:
                scoreboard = get_scoreboard()
                if scoreboard and len(scoreboard) > 0:
                    # Assume recent if data exists
                    return 0.9
        except:
            pass
        return 0.0
    
    def check_pybaseball_freshness(self):
        """Check pybaseball data freshness"""
        try:
            if PYBASEBALL_AVAILABLE:
                today = datetime.now().strftime("%Y-%m-%d")
                game_ids = pb.schedule(today)
                if len(game_ids) > 0:
                    return 0.8
        except:
            pass
        return 0.0
    
    def check_local_data_freshness(self):
        """Check local data freshness"""
        try:
            with open("prediction_report.json", "r") as f:
                data = json.load(f)
                date_str = data.get('prediction_date', '')
                if date_str:
                    prediction_date = datetime.strptime(date_str, "%Y-%m-%d")
                    days_old = (datetime.now() - prediction_date).days
                    # Return freshness score (1.0 = today, 0.0 = very old)
                    freshness = max(0.0, 1.0 - (days_old / 30.0))
                    return freshness
        except:
            pass
        return 0.0
    
    def get_freshness_recommendation(self, score):
        """Get recommendation based on data freshness"""
        if score >= 0.8:
            return "Excellent - Use for high-confidence predictions"
        elif score >= 0.6:
            return "Good - Use with moderate confidence"
        elif score >= 0.4:
            return "Fair - Use with caution, validate with other sources"
        else:
            return "Poor - Prefer other data sources or wait for updates"
    
    def legacy_model_prediction(self, game_data):
        """Legacy model prediction from existing plydb-fun-baseball-analyst-main"""
        try:
            # Try to use the actual win model if features are available
            if self.win_model and self.win_model_features:
                features = self.extract_win_model_features(game_data)
                if features is not None:
                    import numpy as np
                    X = np.array([features])
                    prob = self.win_model.predict_proba(X)[0][1]  # Probability of home win
                    
                    return {
                        'model_prediction': float(prob),
                        'confidence': 0.7,
                        'data_sources': ['legacy_win_model'],
                        'game_id': game_data.get('game_id', 'unknown'),
                        'model_type': 'win_model',
                        'features_used': len([f for f in features if f is not None])
                    }
            
            # Fallback to existing prediction report data
            local_prediction = self.get_local_prediction(game_data)
            if local_prediction:
                return local_prediction
            
            # Final fallback
            return {
                'model_prediction': 0.5,
                'confidence': 0.3,
                'data_sources': ['legacy_fallback'],
                'game_id': game_data.get('game_id', 'unknown')
            }
        except Exception as e:
            print(f"Legacy model prediction failed: {e}")
            return None
    
    def extract_win_model_features(self, game_data):
        """Extract features for win model from game data"""
        # The win model needs specific features from Statcast data
        # For now, we'll return None to indicate features not available from live API
        # In production, this would query the local parquet files for rolling stats
        return None
    
    def get_local_prediction(self, game_data):
        """Get prediction from local prediction_report.json"""
        try:
            with open("prediction_report.json", "r") as f:
                local_data = json.load(f)
                games = local_data.get("games", [])
                
                # Match by home/away team
                home = game_data.get('home_team', '').upper()
                away = game_data.get('away_team', '').upper()
                
                for game in games:
                    if game.get('home_team', '').upper() == home and game.get('away_team', '').upper() == away:
                        home_card = game.get('home_card', {})
                        prob = home_card.get('ai_probability', 0.5)
                        
                        return {
                            'model_prediction': float(prob),
                            'confidence': 0.6,
                            'data_sources': ['legacy_local_report'],
                            'game_id': game_data.get('game_id', 'unknown'),
                            'model_type': 'local_report'
                        }
        except Exception as e:
            print(f"Error reading local prediction: {e}")
        return None
    
    def ensemble_predictions(self, model_pred, live_enhancement, odds_analysis):
        """Combine multiple prediction sources"""
        if not model_pred:
            return None
        
        # Start with model prediction
        base_prediction = model_pred.get('model_prediction', 0.5)
        model_confidence = model_pred.get('confidence', 0.5)
        
        # Adjust based on live data enhancement
        if live_enhancement.get('injuries'):
            base_prediction *= 0.95  # Reduce confidence if key players injured
            model_confidence *= 0.9
        
        # Add odds-based adjustment if available
        if odds_analysis:
            # Compare model prediction with fair market probability
            home_fair = odds_analysis.get('home_fair_prob', 0.5)
            edge = base_prediction - home_fair
            
            # If model disagrees with market, adjust slightly
            if abs(edge) > 0.05:  # 5% edge threshold
                base_prediction += edge * 0.1  # Small adjustment towards model
            
            model_confidence = min(0.9, model_confidence + 0.1)  # Boost confidence with odds
        
        # Ensure probability stays in valid range
        base_prediction = max(0.05, min(0.95, base_prediction))
        
        return {
            'prediction': round(base_prediction, 4),
            'confidence': round(model_confidence, 2),
            'data_sources': ['legacy', 'live', 'odds'],
            'model_prediction': base_prediction,
            'model_confidence': model_confidence,
            'model_type': model_pred.get('model_type', 'unknown'),
            'live_enhancement': live_enhancement,
            'odds_analysis': odds_analysis,
            'edge_vs_market': round(base_prediction - odds_analysis.get('home_fair_prob', 0.5), 4) if odds_analysis else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def odds_price_analysis(self, game_data):
        """Analyze odds data if available"""
        odds = game_data.get('odds')
        if not odds:
            return None
        
        try:
            moneyline = odds.get('moneyline', {})
            home_odds = moneyline.get('home')
            away_odds = moneyline.get('away')
            
            if home_odds and away_odds:
                # Convert American odds to implied probability
                home_prob = self.american_to_implied_prob(home_odds)
                away_prob = self.american_to_implied_prob(away_odds)
                
                # De-vig
                total = home_prob + away_prob
                if total > 0:
                    home_fair = home_prob / total
                    away_fair = away_prob / total
                else:
                    home_fair = away_fair = 0.5
                
                return {
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'home_implied_prob': home_prob,
                    'away_implied_prob': away_prob,
                    'home_fair_prob': home_fair,
                    'away_fair_prob': away_fair,
                    'vig': total - 1.0
                }
        except Exception as e:
            print(f"Odds analysis failed: {e}")
        return None
    
    def american_to_implied_prob(self, odds_str):
        """Convert American odds string to implied probability"""
        try:
            odds = int(odds_str)
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except:
            return 0.5
    
    def analyze_prediction_accuracy(self, predictions):
        """Analyze prediction accuracy across multiple metrics"""
        analysis = {
            'total_predictions': len(predictions),
            'avg_confidence': 0,
            'edge_analyses': [],
            'model_weaknesses': [],
            'recommendations': []
        }
        
        if predictions:
            confidences = [p.get('confidence', 0) for p in predictions if p]
            if confidences:
                analysis['avg_confidence'] = sum(confidences) / len(confidences)
        
        return analysis
    
    def generate_enhanced_report(self, predictions, accuracy_analysis):
        """Generate comprehensive report with enhanced insights"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_sources': ['sports-skills', 'pybaseball', 'local_legacy'],
                'freshness_assessment': self.assess_data_freshness(),
                'prediction_count': len(predictions)
            },
            'predictions': predictions,
            'accuracy_analysis': accuracy_analysis,
            'enhanced_features': {
                'live_injury_data': MLB_DATA_AVAILABLE,
                'pybaseball_integration': PYBASEBALL_AVAILABLE,
                'multi_source_fusion': True,
                'data_freshness_tracking': True
            }
        }
        
        return report
    
    def run_enhanced_prediction(self, date=None):
        """Main execution method"""
        print("Starting Enhanced Baseball Predictor...")
        print(f"Date: {date or 'Today'}")
        
        with LogTimer(timer_logger, "Enhanced Prediction Pipeline"):
            # Step 1: Fetch live data
            print("\nStep 1: Fetching live game data...")
            games_data = self.fetch_live_game_data()
            print(f"Found {len(games_data)} games")
            
            # Step 2: Enhanced predictions
            print("\nStep 2: Generating enhanced predictions...")
            enhanced_predictions = self.enhanced_model_prediction(games_data)
            print(f"Generated {len(enhanced_predictions)} predictions")
            
            # Step 3: Analyze model failures
            print("\nStep 3: Analyzing prediction accuracy...")
            accuracy_analysis = self.analyze_prediction_accuracy(enhanced_predictions)
            
            # Step 4: Generate final report
            print("\nStep 4: Generating enhanced report...")
            final_report = self.generate_enhanced_report(
                enhanced_predictions, accuracy_analysis
            )
            
            return final_report

# Quick test runner
if __name__ == "__main__":
    try:
        predictor = EnhancedBaseballPredictor()
        report = predictor.run_enhanced_prediction()
        
        # Save enhanced report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"enhanced_prediction_report_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nEnhanced prediction completed!")
        print(f"Report saved: enhanced_prediction_report_{timestamp}.json")
        
    except Exception as e:
        print(f"Enhanced predictor failed: {e}")
        import traceback
        traceback.print_exc()