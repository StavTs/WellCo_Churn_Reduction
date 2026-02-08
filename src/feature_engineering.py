"""
Feature engineering module for WellCo churn reduction pipeline.

This module extracts meaningful features from:
- App usage logs
- Member information
- Claims/diagnoses
- Web visits

It also creates interaction features and merges them into a single feature matrix.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Feature engineering for churn prediction and uplift modeling."""

    def __init__(self, observation_end_date: str = '2025-07-14'):
        self.obs_end = pd.to_datetime(observation_end_date)

    # App Usage Features
    def extract_app_usage_features(self, app_usage):
        app_usage['date'] = app_usage['timestamp'].dt.normalize()
        daily_sessions = (
            app_usage.groupby(['member_id', 'date'])
            .size()
            .reset_index(name='sessions')
        )

        first_date = daily_sessions['date'].min()
        last_date = daily_sessions['date'].max()
        window_days = (last_date - first_date).days + 1

        features = {}

        # Basic activity
        total_sessions = daily_sessions.groupby('member_id')['sessions'].sum()
        active_days = daily_sessions.groupby('member_id')['date'].nunique()
        features['total_sessions'] = total_sessions
        features['active_days'] = active_days
        features['avg_sessions_per_day'] = total_sessions / window_days
        features['pct_days_active'] = active_days / window_days
        features['max_sessions_single_day'] = daily_sessions.groupby('member_id')['sessions'].max()
        features['days_since_last_session'] = (last_date - daily_sessions.groupby('member_id')['date'].max()).dt.days

        # Recent activity (last 7 days)
        recent_sessions = daily_sessions[daily_sessions['date'] > (last_date - pd.Timedelta(days=7))]
        features['sessions_last_7d'] = recent_sessions.groupby('member_id')['sessions'].sum()

        # Week 1 vs Week 2 activity ratio
        week_1_end = first_date + pd.Timedelta(days=6)
        week_1 = daily_sessions[daily_sessions['date'] <= week_1_end]
        week_2 = daily_sessions[daily_sessions['date'] > week_1_end]
        week_1_sessions = week_1.groupby('member_id')['sessions'].sum()
        week_2_sessions = week_2.groupby('member_id')['sessions'].sum()
        features['week2_vs_week1_ratio'] = week_2_sessions / week_1_sessions.replace(0, np.nan)

        features_df = pd.DataFrame(features).fillna(0)
        return features_df

    # Member Features
    def extract_member_features(self, members_df, is_test: False):
        features = {}
        features['days_since_signup'] = (self.obs_end - members_df['signup_date']).dt.days
        features['is_new_member'] = (features['days_since_signup'] <= 30).astype(int)
        features['is_mature_member'] = (features['days_since_signup'] >= 365).astype(int)

        features_df = pd.DataFrame({k: v.values for k, v in features.items()}, index=members_df['member_id'])
        labels_df = None
        if not is_test:
            labels_df = members_df[['member_id', 'churn', 'outreach']].set_index('member_id')

        return features_df, labels_df

    #  Claims Features 
    def extract_claims_features(self, claims):
        features = {}
        features['total_claims'] = claims.groupby('member_id').size()
        features['distinct_diagnoses'] = claims.groupby('member_id')['icd_code'].nunique()

        # Priority conditions
        priority_conditions = {'E11.9': 'diabetes', 'I10': 'hypertension', 'Z71.3': 'dietary_counsel'}
        for icd, name in priority_conditions.items():
            features[f'has_{name}'] = (claims[claims['icd_code'] == icd].groupby('member_id').size() > 0).astype(int)

        features['has_priority_condition'] = (
            claims[claims['icd_code'].isin(priority_conditions.keys())].groupby('member_id').size() > 0
        ).astype(int)
        features['num_priority_conditions'] = (
            features['has_diabetes'] + features['has_hypertension'] + features['has_dietary_counsel']
        )

        most_recent_dx = claims.groupby('member_id')['diagnosis_date'].max()
        features['days_since_recent_diagnosis'] = (self.obs_end - most_recent_dx).dt.days
        features['has_recent_diagnosis'] = (features['days_since_recent_diagnosis'] <= 30).astype(int)

        first_dx = claims.groupby('member_id')['diagnosis_date'].min()
        features['days_since_first_diagnosis'] = (self.obs_end - first_dx).dt.days

        features_df = pd.DataFrame(features).fillna(0)
        bool_cols = features_df.select_dtypes(include=['bool']).columns
        features_df[bool_cols] = features_df[bool_cols].astype(int)
        return features_df

    #  Web Visits Features 
    def extract_web_visits_features(self, web_visits):
        # Basic Features
        web_visits['date'] = web_visits['timestamp'].dt.normalize()
        web_visits['domain'] = web_visits['url'].str.extract(r"https?://([^/]+)/")
        web_visits['category'] = web_visits['url'].str.extract(r"https?://[^/]+/([^/]+)/")
        web_visits['is_wellco_domain'] = (web_visits['domain'] == 'health.wellco').astype(int)

        health_domains = ['health.wellco', 'care.portal', 'guide.wellness', 'living.better']
        web_visits['is_health_content'] = web_visits['domain'].isin(health_domains).astype(int)

        # Priority content flags
        diabetes_kw = ['diabetes', 'HbA1c', 'blood glucose', 'glycemic', 'blood sugar']
        hypertension_kw = ['hypertension', 'blood pressure']
        nutrition_kw = ['nutrition', 'Mediterranean', 'fiber', 'diet', 'recipes']

        def has_keywords(text, keywords):
            if pd.isna(text):
                return False
            text = str(text).lower()
            return any(k.lower() in text for k in keywords)

        web_visits['is_diabetes_content'] = web_visits.apply(
            lambda r: has_keywords(r['title'], diabetes_kw) or
                      has_keywords(r['description'], diabetes_kw) or
                      r['category'] == 'diabetes', axis=1
        ).astype(int)
        web_visits['is_hypertension_content'] = web_visits.apply(
            lambda r: has_keywords(r['title'], hypertension_kw) or
                      has_keywords(r['description'], hypertension_kw) or
                      r['category'] == 'hypertension', axis=1
        ).astype(int)
        web_visits['is_nutrition_content'] = web_visits.apply(
            lambda r: has_keywords(r['title'], nutrition_kw) or
                      has_keywords(r['description'], nutrition_kw) or
                      r['category'] in (['nutrition', 'recipes', 'weight']), axis=1
        ).astype(int)

        last_date = web_visits['date'].max()
        features = {}
        features['total_web_visits'] = web_visits.groupby('member_id').size()
        features['web_active_days'] = web_visits.groupby('member_id')['date'].nunique()

        health_visits = web_visits[web_visits['is_health_content'] == 1]
        health_counts = health_visits.groupby('member_id').size()
        features['health_content_pct'] = health_counts / features['total_web_visits']

        wellco_health_counts = web_visits[web_visits['is_wellco_domain'] == 1].groupby('member_id').size()
        features['health_wellco_pct_of_health'] = wellco_health_counts / health_counts

        features['diabetes_content_visits'] = web_visits[web_visits['is_diabetes_content'] == 1].groupby('member_id').size()
        features['hypertension_content_visits'] = web_visits[web_visits['is_hypertension_content'] == 1].groupby('member_id').size()
        features['nutrition_content_visits'] = web_visits[web_visits['is_nutrition_content'] == 1].groupby('member_id').size()

        features['days_since_last_web_visit'] = (last_date - web_visits.groupby('member_id')['date'].max()).dt.days

        # Recent 7-day web activity
        recent_7d = web_visits[web_visits['date'] >= (last_date - pd.Timedelta(days=6))]
        features['web_visits_last_7d'] = recent_7d.groupby('member_id').size()

        # Week 1 vs Week 2
        first_date = web_visits['date'].min()
        week_1_end = first_date + pd.Timedelta(days=6)
        week_1 = web_visits[web_visits['date'] <= week_1_end]
        week_2 = web_visits[web_visits['date'] > week_1_end]
        features['web_week2_vs_week1_ratio'] = week_2.groupby('member_id').size() / week_1.groupby('member_id').size().replace(0, np.nan)

        features_df = pd.DataFrame(features).fillna(0).replace([np.inf, -np.inf], 0)
        return features_df

    # Interaction Features 
    def create_interaction_features(self, all_features):
        # Create features from different files
        all_features['has_diabetes_no_diabetes_content'] = (
            (all_features['has_diabetes'] == 1) & (all_features['diabetes_content_visits'] == 0)
        ).astype(int)
        all_features['has_hypertension_no_hypertension_content'] = (
            (all_features['has_hypertension'] == 1) & (all_features['hypertension_content_visits'] == 0)
        ).astype(int)
        all_features['has_dietary_counsel_no_nutrition_content'] = (
            (all_features['has_dietary_counsel'] == 1) & (all_features['nutrition_content_visits'] == 0)
        ).astype(int)

        all_features['priority_condition_low_health_content'] = (
            (all_features['has_priority_condition'] == 1) & (all_features['health_content_pct'] < 0.5)
        ).astype(int)

        all_features['app_and_web_declining'] = (
            (all_features['week2_vs_week1_ratio'] < 1) & (all_features['web_week2_vs_week1_ratio'] < 1)
        ).astype(int)

        all_features['content_alignment_score'] = (
            all_features['has_diabetes'] * all_features['diabetes_content_visits'] +
            all_features['has_hypertension'] * all_features['hypertension_content_visits'] +
            all_features['has_dietary_counsel'] * all_features['nutrition_content_visits']
        ) / (all_features['num_priority_conditions'] + 1)

        all_features['sessions_per_active_day'] = all_features['total_sessions'] / (all_features['active_days'] + 1)
        all_features['recent_engagement_share'] = all_features['sessions_last_7d'] / (all_features['total_sessions'] + 1)

        return all_features

    # Merge All Features 
    def merge_all_features(self, app_features, member_features, claims_features, web_features, reference_index):
        
        all_members = pd.DataFrame(index=reference_index)
        all_members.index.name = 'member_id'

        merged = all_members.join(app_features, how='left')
        merged = merged.join(member_features, how='left')
        merged = merged.join(claims_features, how='left')
        merged = merged.join(web_features, how='left')
        merged = merged.fillna(0)

        # Add interaction features
        merged = self.create_interaction_features(merged)

        print(f"  Final feature set: {merged.shape[1]} features, {merged.shape[0]} members")
        return merged
