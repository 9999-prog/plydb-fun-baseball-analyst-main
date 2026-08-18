"""
Docker CrewAI Integration for MLB Analyst
Allows the main Python 3.14 project to call CrewAI analysis via Docker.
"""
import json
import subprocess
import os
from typing import Dict, Any, Optional
from pathlib import Path


class CrewAIDockerClient:
    """Client for running CrewAI analysis in Docker container."""
    
    def __init__(self, docker_image: str = "crewai-mlb", 
                 docker_dir: Optional[str] = None):
        self.docker_image = docker_image
        self.docker_dir = docker_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "crewai_docker"
        )
        self._ensure_built()
    
    def _ensure_built(self):
        """Build Docker image if not exists."""
        try:
            # Check if image exists
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True, text=True, timeout=10
            )
            if not result.stdout.strip():
                print(f"🔨 Building Docker image: {self.docker_image}")
                self.build()
        except Exception as e:
            print(f"⚠️ Docker check failed: {e}")
    
    def build(self) -> bool:
        """Build the Docker image."""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", self.docker_image, "."],
                cwd=self.docker_dir,
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"✅ Docker image built: {self.docker_image}")
                return True
            else:
                print(f"❌ Docker build failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Docker build timed out")
            return False
        except Exception as e:
            print(f"❌ Docker build error: {e}")
            return False
    
    def analyze_game(self, game_data: Dict[str, Any], 
                     timeout: int = 120) -> Dict[str, Any]:
        """
        Run CrewAI analysis on a game.
        
        Args:
            game_data: Dict with game info (home_team, away_team, model_prob, odds, etc.)
            timeout: Max seconds to wait for analysis
            
        Returns:
            Dict with analysis result or error
        """
        # Prepare input JSON
        input_json = json.dumps(game_data)
        
        try:
            # Run Docker container with stdin input
            result = subprocess.run(
                ["docker", "run", "--rm", "-i", self.docker_image],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return {
                    "error": "Docker container failed",
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            
            # Parse output - last JSON block is the result
            output = result.stdout
            
            # Try to find analysis_result.json in output
            if os.path.exists(os.path.join(self.docker_dir, "analysis_result.json")):
                with open(os.path.join(self.docker_dir, "analysis_result.json"), 'r') as f:
                    return json.load(f)
            
            # Otherwise return raw output
            return {
                "raw_output": output,
                "stderr": result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            return {"error": f"Analysis timed out after {timeout}s"}
        except Exception as e:
            return {"error": f"Docker execution failed: {e}"}
    
    def analyze_batch(self, games: list, timeout: int = 300) -> list:
        """Analyze multiple games sequentially."""
        results = []
        for i, game in enumerate(games):
            print(f"[{i+1}/{len(games)}] Analyzing {game.get('away_team')} @ {game.get('home_team')}...")
            result = self.analyze_game(game, timeout=timeout)
            results.append(result)
        return results


def create_game_data_from_prediction(prediction: Dict) -> Dict:
    """
    Convert enhanced_baseball_predictor output to CrewAI input format.
    """
    return {
        "home_team": prediction.get("home_team", ""),
        "away_team": prediction.get("away_team", ""),
        "model_prob": prediction.get("prediction", 0.5),
        "confidence": prediction.get("confidence", 0.5),
        "model_agreement": prediction.get("model_agreement", 0.5),
        "odds": prediction.get("odds_analysis", {}),
        "individual_models": {
            k: v[0] if isinstance(v, list) else v 
            for k, v in prediction.get("individual_models", {}).items()
        }
    }


# ============================================================
# Integration with Enhanced Predictor
# ============================================================

def run_crewai_analysis_on_predictions(predictions: list) -> list:
    """
    Run CrewAI analysis on a list of predictions from enhanced predictor.
    """
    client = CrewAIDockerClient()
    results = []
    
    for pred in predictions:
        if pred.get("odds_analysis") and pred.get("odds_analysis").get("value_bet"):
            # Only analyze games with betting value
            game_data = create_game_data_from_prediction(pred)
            print(f"🔍 CrewAI analyzing: {game_data['away_team']} @ {game_data['home_team']}")
            result = client.analyze_game(game_data)
            results.append({
                "game": f"{game_data['away_team']} @ {game_data['home_team']}",
                "crewai_analysis": result
            })
    
    return results


if __name__ == "__main__":
    # Test the client
    client = CrewAIDockerClient()
    
    test_game = {
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
    
    print("🧪 Testing CrewAI Docker client...")
    result = client.analyze_game(test_game)
    print(json.dumps(result, indent=2, default=str))