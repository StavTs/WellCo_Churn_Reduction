"""
Evaluation and Outreach Selection Module

This module provides functionality to:
- Rank and select members for outreach based on predicted uplift (CATE) and business priorities.
- Compute weighted treatment effects for outreach optimization.
- Create visualizations to support model evaluation and outreach decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from kneed import KneeLocator


class OutreachSelector:
    def __init__(self, member_values = None):
        self.member_values = member_values

    def compute_priority_score(self, df):
        df = df.copy()
        df['tenure_norm'] = df['days_since_signup'] / df['days_since_signup'].max()

        # Weighted priority scoring
        df['priority_score'] = (
            2.0 * df['has_priority_condition'] +       # Health condition importance
            1.5 * df['health_content_pct'] +          # Engagement with health content
            1.0 * df['tenure_norm']                   # Member tenure
        )
        return df

    def apply_weighted_uplift(self, df):
        df = self.compute_priority_score(df)
        df['weighted_uplift'] = df['cate'] * df['priority_score'] * df['churn_prob_no_outreach']
        return df

    def rank_members(self, treatment_effects):
        df = self.apply_weighted_uplift(treatment_effects.copy())
        df = df.sort_values("weighted_uplift", ascending=False).reset_index()
        df['rank'] = range(1, len(df) + 1)
        return df

    def select_top_n(self, ranked_members, n):
        outreach_list = ranked_members.head(n).copy()
        print(f"Selected top {n} members for outreach")
        return outreach_list

    def optimal_n(self, ranked_members, save_path, cost_per_member = 0.0):
        df = ranked_members.copy()
        df = df.sort_values("weighted_uplift", ascending=False).reset_index(drop=True)
        df['net_value'] = df['weighted_uplift'] - cost_per_member
        df['cumulative_uplift'] = df['net_value'].cumsum()
        x_values = df['rank'].to_numpy()
        y_values = df['cumulative_uplift'].to_numpy()

        kneedle = KneeLocator(x_values, y_values, curve='concave', direction='increasing')
        optimal_n = kneedle.knee

        # Plot cumulative uplift with elbow point
        plt.figure(figsize=(10,6)) 
        plt.plot(x_values, y_values, linewidth=2, label='Cumulative Weighted Uplift') 
        if optimal_n is not None: 
            plt.axvline(optimal_n, color='red', linestyle='--', label=f'Elbow (N={optimal_n})') 
        plt.xlabel('Number of Members Reached') 
        plt.ylabel('Cumulative Weighted Uplift') 
        plt.title('Elbow Method for Optimal Outreach Size') 
        plt.grid(True, alpha=0.3) 
        plt.legend() 
        plt.tight_layout() 
        plt.savefig(save_path, dpi=300) 
        plt.close()

        return optimal_n

    def create_outreach_list(self, ranked_members, top_n):
        # Select top N members
        outreach_list = ranked_members.head(top_n).copy()

        # Keep only relevant columns
        outreach_list = outreach_list[['rank', 'cate', 'churn_prob_no_outreach']]

        # Rename columns for clarity
        outreach_list = outreach_list.rename(columns={
            'cate': 'treatment_effect',
            'churn_prob_no_outreach': 'churn_risk_baseline'
        })

        print(f" Outreach list created: {len(outreach_list):,} members")
        return outreach_list


class Visualizer:


    @staticmethod
    def plot_model_performance(y_train, y_val, y_train_proba, y_val_proba, save_path):
        """
        Plot ROC and Precision-Recall curves for train and validation datasets.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba)
        axes[0].plot(fpr_train, tpr_train, label=f'Train (AUC={roc_auc_score(y_train, y_train_proba):.3f})', linewidth=2)
        axes[0].plot(fpr_val, tpr_val, label=f'Val (AUC={roc_auc_score(y_val, y_val_proba):.3f})', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        axes[1].plot(recall, precision, linewidth=2)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Performance plot saved: {save_path}")

    @staticmethod
    def plot_cate_distribution(treatment_effects, save_path):
        """
        Plot histogram and cumulative plot of CATE distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(treatment_effects['cate_percentage_points'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No effect')
        axes[0].axvline(treatment_effects['cate_percentage_points'].mean(),
                        color='green', linestyle='--', linewidth=2, label='Mean CATE')
        axes[0].set_xlabel('CATE (percentage points)')
        axes[0].set_ylabel('Number of members')
        axes[0].set_title('Distribution of Treatment Effects')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Cumulative CATE
        ranked = treatment_effects.sort_values('cate', ascending=False).copy()
        ranked['rank'] = range(1, len(ranked) + 1)
        ranked['cumulative_cate'] = ranked['cate'].cumsum()
        axes[1].plot(ranked['rank'].to_numpy(), ranked['cumulative_cate'].to_numpy(), linewidth=2)
        axes[1].set_xlabel('Number of Members Reached')
        axes[1].set_ylabel('Cumulative Treatment Effect')
        axes[1].set_title('Cumulative Expected Churns Prevented')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  CATE distribution plot saved: {save_path}")

