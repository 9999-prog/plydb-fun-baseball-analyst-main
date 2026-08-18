#!/usr/bin/env python3
"""
Train win_model.joblib - Team win probability model

Creates game-level features from statcast_multiseason_pa_level_model_ready.parquet
Target: home_team wins (1) or loses (0)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, using GradientBoostingClassifier")

# Paths
PA_DATA_PATH = Path(__file__).parent.parent / "data" / "pybaseball" / "statcast" / "statcast_multiseason_pa_level_model_ready.parquet"
MODEL_PATH = Path(__file__).parent / "win_model.joblib"