"""
Data loading and validation utilities.

Provides functions to:
1. Load training and test datasets with proper date parsing.
2. Perform basic data quality checks (missing values, duplicates, shape).

Designed for a churn prediction and outreach pipeline.
"""

import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

# Load Data Functions
def load_training_data(app_path = None, labels_path = None, claims_path = None, web_path = None):
    if app_path == None:
        from .config import TRAIN_APP_USAGE, TRAIN_CHURN_LABELS, TRAIN_CLAIMS, TRAIN_WEB_VISITS
    else:
        TRAIN_APP_USAGE = app_path
        TRAIN_CHURN_LABELS = labels_path
        TRAIN_CLAIMS = claims_path
        TRAIN_WEB_VISITS = web_path

    print("Loading training data...")

    # Load datasets with datetime parsing
    app_usage = pd.read_csv(TRAIN_APP_USAGE, parse_dates=["timestamp"])
    churn_labels = pd.read_csv(TRAIN_CHURN_LABELS, parse_dates=["signup_date"])
    claims = pd.read_csv(TRAIN_CLAIMS, parse_dates=["diagnosis_date"])
    web_visits = pd.read_csv(TRAIN_WEB_VISITS, parse_dates=["timestamp"])

    # Print basic info for verification
    print(f"App usage: {len(app_usage):,} rows")
    print(f"Churn labels: {len(churn_labels):,} rows")
    print(f"Claims: {len(claims):,} rows")
    print(f"Web visits: {len(web_visits):,} rows")

    return app_usage, churn_labels, claims, web_visits


def load_test_data(test_app_path = None, test_claims_path = None, test_members_path = None, test_web_path = None):

    if test_app_path == None:
        from .config import TEST_APP_USAGE, TEST_CLAIMS, TEST_MEMBERS, TEST_WEB_VISITS
    else:
        TEST_APP_USAGE = test_app_path
        TEST_CLAIMS = test_claims_path
        TEST_MEMBERS = test_members_path
        TEST_WEB_VISITS = test_web_path

    print("Loading test data...")

    app_usage = pd.read_csv(TEST_APP_USAGE, parse_dates=["timestamp"])
    claims = pd.read_csv(TEST_CLAIMS, parse_dates=["diagnosis_date"])
    members = pd.read_csv(TEST_MEMBERS, parse_dates=["signup_date"])
    web_visits = pd.read_csv(TEST_WEB_VISITS, parse_dates=["timestamp"])

    print(f"App usage: {len(app_usage):,} rows")
    print(f"Members: {len(members):,} rows")
    print(f"Claims: {len(claims):,} rows")
    print(f"Web visits: {len(web_visits):,} rows")

    return app_usage, claims, members, web_visits


# Data Validation Function
def validate_data_quality(df, name):
    print(f"\n{name} quality check:")
    print(f"Shape: {df.shape}")

    # Identify columns with missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print("Missing values detected:")
        print(missing)
    else:
        print("No missing values detected.")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")
