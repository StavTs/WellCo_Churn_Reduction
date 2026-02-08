"""
S-Learner Model for WellCo Churn Reduction Pipeline.

This module implements an S-Learner approach for estimating individualized treatment effects (CATE)
using XGBoost. It includes methods to:

- Prepare training and validation data with stratified splits
- Train the model and evaluate performance
- Estimate CATE for each member under outreach vs no outreach
- Save and load trained models with feature names
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib


class SLearnerModel:
    def __init__(self, params, random_state = 42):
        self.params = params
        self.random_state = random_state
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] | None = None

    def prepare_data( self, features, labels, test_size = 0.2):
        # Add 'outreach' as a feature for S-Learner
        training_data = features.copy()
        training_data['outreach'] = labels['outreach']

        # Target variable
        y = labels['churn']

        # Keep feature list for later prediction
        self.feature_columns = training_data.columns.tolist()
        X = training_data[self.feature_columns]

        # Compute scale_pos_weight for imbalanced classification
        churn_count = (y == 1).sum()
        non_churn_count = (y == 0).sum()
        self.params['scale_pos_weight'] = non_churn_count / churn_count

        # Stratified split to maintain class distribution
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        return X_train, X_val, y_train, y_val

    def train( self, X_train, y_train, X_val, y_val):

        self.model = XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Predictions and AUC evaluation
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        val_auc = roc_auc_score(y_val, y_val_proba)

        print(" ")
        print("Model Performance")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Overfitting gap: {train_auc - val_auc:.4f}")

        # Classification report on validation set
        y_val_pred = self.model.predict(X_val)
        print("\nClassification Report (Validation):")
        print(classification_report(y_val, y_val_pred, target_names=['Non-Churn', 'Churn']))

        return self.model

    def estimate_cate(self, features):
        print("\nEstimating treatment effects (CATE)...")
        X = features[self.feature_columns].copy()

        # Scenario 1: no outreach
        X_no_outreach = X.copy()
        X_no_outreach['outreach'] = 0
        prob_churn_no_outreach = self.model.predict_proba(X_no_outreach)[:, 1]

        # Scenario 2: with outreach
        X_with_outreach = X.copy()
        X_with_outreach['outreach'] = 1
        prob_churn_with_outreach = self.model.predict_proba(X_with_outreach)[:, 1]

        # Calculate CATE
        cate = prob_churn_no_outreach - prob_churn_with_outreach

        results = pd.DataFrame({
            'member_id': features.index,
            'churn_prob_no_outreach': prob_churn_no_outreach,
            'churn_prob_with_outreach': prob_churn_with_outreach,
            'cate': cate,
            'cate_percentage_points': cate * 100
        }).set_index('member_id')

        print(f"  Mean CATE: {results['cate'].mean():.4f}")
        print(f"  Median CATE: {results['cate'].median():.4f}")
        print(f"  % Positive CATE: {(results['cate'] > 0).mean():.1%}")

        return results

    def save_model(self, model_path, feature_names_path):
        joblib.dump(self.model, model_path)
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))

        print(f"\nModel saved to: {model_path}")
        print(f"Feature names saved to: {feature_names_path}")

    def load_model(self, model_path, feature_names_path):
        self.model = joblib.load(model_path)
        with open(feature_names_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f]

        print(f"Model loaded from: {model_path}")
