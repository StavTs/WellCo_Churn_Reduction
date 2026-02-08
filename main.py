"""
Main execution pipeline for the WellCo churn-reduction solution.
"""

import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import *
from src.utils import load_training_data, load_test_data
from src.feature_engineering import FeatureEngineer
from src.model import SLearnerModel
from src.evaluation import OutreachSelector, Visualizer


def run_training_pipeline():
    """
    Execute the full training pipeline:
    - Load data
    - Engineer features
    - Train S-Learner model
    - Evaluate performance
    - Estimate training CATE
    """
    print("Training Pipeline")
    print(" ")

    # Load training data 
    app_usage, churn_labels, claims, web_visits = load_training_data()

    # Feature Engeeniring
    engineer = FeatureEngineer(OBSERVATION_END_DATE)

    app_features = engineer.extract_app_usage_features(app_usage)
    member_features, labels = engineer.extract_member_features(churn_labels, is_test=False)
    claims_features = engineer.extract_claims_features(claims)
    web_features = engineer.extract_web_visits_features(web_visits)

    all_features = engineer.merge_all_features(
        app_features,
        member_features,
        claims_features,
        web_features,
        reference_index=labels.index
    )

    print(" ")
    print(f"Training feature matrix shape: {all_features.shape}")
    print(f"Churn rate: {labels['churn'].mean():.2%}")
    print(" ")

    # Train model using single learner
    s_learner = SLearnerModel(XGBOOST_PARAMS, RANDOM_STATE)
    X_train, X_val, y_train, y_val = s_learner.prepare_data(all_features, labels, TEST_SIZE)
    model = s_learner.train(X_train, y_train, X_val, y_val)

    # Save trained model
    s_learner.save_model(MODEL_FILE, FEATURE_NAMES_FILE)

    # Model Evaluation
    viz = Visualizer()

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    viz.plot_model_performance(
        y_train,
        y_val,
        y_train_proba,
        y_val_proba,
        os.path.join(FIGURES_DIR, "model_performance.png")
    )

    # Training CATE
    all_features_with_outreach = all_features.copy()
    all_features_with_outreach["outreach"] = labels["outreach"]

    treatment_effects_train = s_learner.estimate_cate(all_features_with_outreach)

    viz.plot_cate_distribution(
        treatment_effects_train,
        os.path.join(FIGURES_DIR, "cate_distribution_train.png")
    )

    return engineer, s_learner


def run_test_pipeline(engineer, s_learner):
    """
    Apply trained model to test population:
    - Build test features
    - Estimate treatment effects
    - Rank members by uplift
    - Generate outreach list
    """
    print(" ")
    print("Test Application")
    print(" ")
    # Load test data
    test_app_usage, test_claims, test_members, test_web_visits = load_test_data()

    # Feature engineering on the test data
    test_app_features = engineer.extract_app_usage_features(test_app_usage)
    test_member_features, _ = engineer.extract_member_features(test_members, is_test=True)
    test_claims_features = engineer.extract_claims_features(test_claims)
    test_web_features = engineer.extract_web_visits_features(test_web_visits)

    test_all_features = engineer.merge_all_features(
        test_app_features,
        test_member_features,
        test_claims_features,
        test_web_features,
        reference_index=test_members.set_index("member_id").index
    )

    test_all_features["outreach"] = 0

    # Estimate test CATE
    treatment_effects_test = s_learner.estimate_cate(test_all_features)

    selector = OutreachSelector()

    treatment_effects_test = treatment_effects_test.join(
        test_all_features[["days_since_signup", "has_priority_condition", "health_content_pct"]]
    )

    treatment_effects_test = selector.apply_weighted_uplift(treatment_effects_test)
    ranked_members = treatment_effects_test.sort_values("weighted_uplift", ascending=False)
    ranked_members["rank"] = range(1, len(ranked_members) + 1)

    optimal_n = selector.optimal_n(
        ranked_members,
        os.path.join(FIGURES_DIR, "optimal_n_elbow.png")
    )

    outreach_list = selector.select_top_n(ranked_members, n=optimal_n)

    outreach_list.to_csv(OUTREACH_LIST_FILE, index=False)
    print(f"Outreach list saved: {OUTREACH_LIST_FILE}")

    return outreach_list


def main():
    """Run the end-to-end churn reduction pipeline."""
    print(" ")
    print("WellCo Churn Reduction Pipeline")
    print(f"Start time: {datetime.now()}")
    print(" ")
    engineer, s_learner = run_training_pipeline()
    outreach_list = run_test_pipeline(engineer, s_learner)
    print(" ")
    print("Pipline Complete")
    print(f"Members selected for outreach: {len(outreach_list):,}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Model saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
