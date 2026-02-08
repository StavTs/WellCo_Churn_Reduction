"""
Configuration file for WellCo churn reduction pipeline.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Training Data Paths 
TRAIN_APP_USAGE = os.path.join(TRAIN_DIR, "app_usage.csv")
TRAIN_CHURN_LABELS = os.path.join(TRAIN_DIR, "churn_labels.csv")
TRAIN_CLAIMS = os.path.join(TRAIN_DIR, "claims.csv")
TRAIN_WEB_VISITS = os.path.join(TRAIN_DIR, "web_visits.csv")

# Test Data Paths 
TEST_APP_USAGE = os.path.join(TEST_DIR, "test_app_usage.csv")
TEST_CLAIMS = os.path.join(TEST_DIR, "test_claims.csv")
TEST_MEMBERS = os.path.join(TEST_DIR, "test_members.csv")
TEST_WEB_VISITS = os.path.join(TEST_DIR, "test_web_visits.csv")

# Output Files 
OUTREACH_LIST_FILE = os.path.join(OUTPUT_DIR, "outreach_list.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "s_learner_model.pkl")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.pkl")
FEATURE_NAMES_FILE = os.path.join(OUTPUT_DIR, "feature_names.txt")

# Date
OBSERVATION_END_DATE = "2025-07-14"

# Model Parametes
RANDOM_STATE = 42
TEST_SIZE = 0.2

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5,
    "random_state": 42
}