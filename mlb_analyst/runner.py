#!/usr/bin/env python3
"""
Optimized runner for the enhanced baseball predictor
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

def main():
    """Main execution with error handling and performance monitoring"""
    
    print("Enhanced MLB Baseball Predictor")
    print("=" * 50)
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import and run the enhanced predictor
        from enhanced_baseball_predictor import EnhancedBaseballPredictor
        
        predictor = EnhancedBaseballPredictor()
        report = predictor.run_enhanced_prediction()
        
        print("\nPrediction completed successfully!")
        
        # Display key insights
        if report and 'predictions' in report:
            preds = report['predictions']
            print(f"\nGenerated {len(preds)} predictions:")
            
            for i, p in enumerate(preds):
                model_type = p.get('model_type', 'unknown')
                pred = p['prediction']
                conf = p['confidence']
                edge = p.get('edge_vs_market')
                odds = p.get('odds_analysis')
                
                print(f"  Game {i+1}: {model_type:15} | pred={pred:.3f} | conf={conf:.2f}", end="")
                if edge is not None:
                    print(f" | edge={edge:+.3f}", end="")
                if odds:
                    print(f" | odds: H{odds['home_odds']}/A{odds['away_odds']}", end="")
                print()
        
        # Freshness assessment
        if report and 'metadata' in report:
            freshness = report['metadata'].get('freshness_assessment', {})
            print(f"\nData Freshness: {freshness.get('recommendation', 'Unknown')}")
            print(f"Overall Score: {freshness.get('overall_score', 0):.2f}")
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        print("Try installing missing packages:")
        print("  pip install sports-skills pybaseball requests psutil tqdm matplotlib")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()