"""
CrewAI MLB Analysis Agent
Runs in Docker container, analyzes baseball predictions from the main system.
"""
import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append('/app')

# ============================================================
# TOOLS
# ============================================================

class OddsAnalysisTool(BaseTool):
    """Analyze betting odds and find value."""
    name: str = "odds_analysis"
    description: str = "Analyze moneyline odds, calculate implied probability, and detect value bets"
    
    def _run(self, home_odds: str, away_odds: str) -> str:
        """Analyze American odds and return fair probabilities."""
        def american_to_decimal(odds_str: str) -> float:
            odds = int(odds_str)
            if odds > 0:
                return 1 + odds / 100
            return 1 + 100 / abs(odds)
        
        home_dec = american_to_decimal(home_odds)
        away_dec = american_to_decimal(away_odds)
        
        implied_home = 1 / home_dec
        implied_away = 1 / away_dec
        total = implied_home + implied_away
        
        fair_home = implied_home / total
        fair_away = implied_away / total
        vig = total - 1.0
        
        return json.dumps({
            "home_odds": home_odds,
            "away_odds": away_odds,
            "implied_home_prob": round(implied_home, 4),
            "implied_away_prob": round(implied_away, 4),
            "fair_home_prob": round(fair_home, 4),
            "fair_away_prob": round(fair_away, 4),
            "vig": round(vig, 4)
        })


class EdgeDetectionTool(BaseTool):
    """Detect edge between model prediction and market odds."""
    name: str = "edge_detection"
    description: str = "Compare model probability with fair market probability to find betting edge"
    
    def _run(self, model_prob: float, fair_market_prob: float) -> str:
        """Calculate edge and Kelly criterion."""
        edge = model_prob - fair_market_prob
        
        # Kelly criterion (fraction of bankroll)
        if model_prob > 0 and fair_market_prob > 0:
            decimal_odds = 1 / fair_market_prob
            kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
            kelly = max(0, min(kelly, 0.25))  # Cap at 25% Kelly
        else:
            kelly = 0
        
        recommendation = "BET" if edge > 0.02 else "NO_BET"
        if edge > 0.05:
            recommendation = "STRONG_BET"
        elif edge < -0.02:
            recommendation = "MARKET_DISAGREES"
        
        return json.dumps({
            "model_probability": round(model_prob, 4),
            "fair_market_probability": round(fair_market_prob, 4),
            "edge": round(edge, 4),
            "edge_percent": f"{edge*100:.1f}%",
            "kelly_fraction": round(kelly, 4),
            "recommendation": recommendation
        })


class TeamStatsTool(BaseTool):
    """Look up team stats from the blended dataset."""
    name: str = "team_stats"
    description: str = "Get latest team offensive/defensive stats from 71K game dataset"
    
    def _run(self, team: str) -> str:
        """Return team stats (mock - would connect to actual data)."""
        # In production, this would query the blended_stats.csv
        return json.dumps({
            "team": team,
            "offensive": {"woba": 0.330, "k_pct": 0.22, "bb_pct": 0.08},
            "defensive": {"era": 3.80, "k_per_9": 9.0, "bb_per_9": 3.2}
        })


# ============================================================
# AGENTS
# ============================================================

def create_agents():
    """Create the CrewAI agents for MLB analysis."""
    
    # Data Analyst Agent
    data_analyst = Agent(
        role='MLB Data Analyst',
        goal='Analyze team statistics, Statcast metrics, and historical performance',
        backstory="""You are an expert MLB data analyst with deep knowledge of 
        Statcast metrics (xwOBA, hard-hit%, barrel rate, spin rate, velocity), 
        team offensive/defensive splits, and park factors. You can interpret 
        the 96-feature dataset from the MLB-Game-Winner-Predictor project.""",
        verbose=True,
        allow_delegation=False,
        tools=[TeamStatsTool()]
    )
    
    # Odds Analyst Agent
    odds_analyst = Agent(
        role='Sports Betting Odds Analyst',
        goal='Analyze betting markets, calculate fair odds, and detect value',
        backstory="""You are a professional sports betting analyst. You expertly 
        convert American odds to implied probabilities, remove vig to find fair 
        market prices, and identify when a model has an edge over the market. 
        You understand Kelly criterion for optimal bet sizing.""",
        verbose=True,
        allow_delegation=False,
        tools=[OddsAnalysisTool(), EdgeDetectionTool()]
    )
    
    # Prediction Strategist Agent
    strategist = Agent(
        role='MLB Prediction Strategist',
        goal='Synthesize data analysis and odds analysis into actionable predictions',
        backstory="""You are the lead strategist who combines quantitative model 
        outputs with market analysis. You weigh model confidence, market edge, 
        model agreement, and risk factors to make final recommendations. 
        You understand ensemble methods (Gradient Boosting, XGBoost, MLP, SVC, NearestCentroid) 
        and how to interpret their consensus.""",
        verbose=True,
        allow_delegation=True
    )
    
    return data_analyst, odds_analyst, strategist


# ============================================================
# TASKS
# ============================================================

def create_tasks(data_analyst, odds_analyst, strategist, game_data: Dict):
    """Create tasks for the crew based on game data."""
    
    home = game_data.get('home_team', '')
    away = game_data.get('away_team', '')
    model_prob = game_data.get('model_prob', 0.5)
    home_odds = game_data.get('odds', {}).get('moneyline', {}).get('home', '')
    away_odds = game_data.get('odds', {}).get('moneyline', {}).get('away', '')
    
    task1 = Task(
        description=f"""
        Analyze the matchup between {home} (home) and {away} (away).
        
        Retrieve and analyze:
        1. Offensive stats for both teams (wOBA, K%, BB%, hard-hit%, barrels/PA)
        2. Defensive/pitching stats (ERA, K/9, BB/9, HR/9, velocity, spin rate)
        3. Home/away splits and park factors
        4. Recent form (last 10 games)
        5. Head-to-head history
        
        Provide a summary of which team has the statistical advantage and why.
        """,
        agent=data_analyst,
        expected_output="Detailed statistical analysis report with team comparisons"
    )
    
    task2 = Task(
        description=f"""
        Analyze the betting market for {home} vs {away}.
        
        Given odds: Home {home_odds}, Away {away_odds}
        
        1. Convert American odds to decimal and implied probabilities
        2. Remove vig to calculate fair market probabilities
        3. Calculate the bookmaker's vig/overround
        4. Identify if either side has value
        
        Output the fair home/away probabilities and market assessment.
        """,
        agent=odds_analyst,
        expected_output="Market analysis with fair probabilities and vig calculation"
    )
    
    task3 = Task(
        description=f"""
        Detect edge between model prediction and market.
        
        Model probability (home win): {model_prob:.2%}
        
        Using the fair market probabilities from Task 2:
        1. Calculate edge = model_prob - fair_market_prob
        2. Calculate Kelly criterion bet size (cap at 25% Kelly)
        3. Provide recommendation: STRONG_BET (>5% edge), BET (>2% edge), NO_BET, or MARKET_DISAGREES
        
        Output the edge analysis with specific recommendation.
        """,
        agent=odds_analyst,
        expected_output="Edge analysis with Kelly sizing and betting recommendation"
    )
    
    task4 = Task(
        description=f"""
        Synthesize all analysis into final prediction for {home} vs {away}.
        
        Consider:
        1. Statistical advantage from Task 1
        2. Market fair probabilities from Task 2
        3. Model edge and recommendation from Task 3
        4. Model confidence: {game_data.get('confidence', 0.5):.2%}
        5. Model agreement: {game_data.get('model_agreement', 0.5):.2%}
        6. Individual model views: {game_data.get('individual_models', {})}
        
        Provide:
        - Final predicted winner
        - Confidence level (0-100%)
        - Betting recommendation with stake size
        - Key risk factors
        - Reasoning summary
        """,
        agent=strategist,
        expected_output="Final prediction report with winner, confidence, bet recommendation, and reasoning"
    )
    
    return [task1, task2, task3, task4]


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point - reads game data from stdin or file."""
    
    # Read input (JSON from stdin or file)
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            game_data = json.load(f)
    else:
        # Read from stdin
        input_data = sys.stdin.read()
        if input_data:
            game_data = json.loads(input_data)
        else:
            # Default test game
            game_data = {
                "home_team": "LAD",
                "away_team": "NYY",
                "model_prob": 0.784,
                "confidence": 0.784,
                "model_agreement": 1.0,
                "odds": {
                    "moneyline": {
                        "home": "-150",
                        "away": "+130"
                    }
                },
                "individual_models": {
                    "gradient_boosting": 0.779,
                    "xgboost": 0.696,
                    "mlp": 0.999,
                    "svc": 0.517,
                    "nearest_centroid": 0.990
                }
            }
    
    print(f"🎯 Analyzing: {game_data['away_team']} @ {game_data['home_team']}")
    print(f"📊 Model prob: {game_data['model_prob']:.1%}")
    if game_data.get('odds'):
        odds = game_data['odds'].get('moneyline', {})
        print(f"💰 Odds: Home {odds.get('home')} / Away {odds.get('away')}")
    print()
    
    # Create agents and tasks
    data_analyst, odds_analyst, strategist = create_agents()
    tasks = create_tasks(data_analyst, odds_analyst, strategist, game_data)
    
    # Create crew
    crew = Crew(
        agents=[data_analyst, odds_analyst, strategist],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    # Run analysis
    print("🚀 Starting CrewAI analysis...")
    print("=" * 60)
    result = crew.kickoff()
    
    print("\n" + "=" * 60)
    print("📋 FINAL REPORT")
    print("=" * 60)
    print(result)
    
    # Save result
    output = {
        "timestamp": datetime.now().isoformat(),
        "game": f"{game_data['away_team']} @ {game_data['home_team']}",
        "analysis": str(result)
    }
    
    with open('/app/analysis_result.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✅ Analysis saved to /app/analysis_result.json")


if __name__ == "__main__":
    main()