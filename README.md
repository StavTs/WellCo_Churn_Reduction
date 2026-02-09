# WellCo Churn Reduction & Outreach Optimization
## Overview

WellCo is experiencing increasing member churn and aims to proactively retain members through targeted outreach.
This project develops an end-to-end churn reduction pipeline that:

- Predicts churn risk

- Estimates the causal impact of outreach using uplift modeling (S-Learner)

- Produces a ranked list of members for prioritized outreach

- Determines the optimal outreach size (n) based on incremental value

- The final output is a CSV containing the top prioritized members for outreach along with their prioritization scores and rank.

The goal is not only to predict churn, but to **identify which members should be contacted to maximize retention impact**.

Traditional churn prediction prioritizes high-risk members, while this solution prioritizes **members whose churn probability is expected to decrease the most if outreach occurs**, ensuring outreach resources are allocated where they generate the greatest value.


## Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/StavTs/WellCo_Churn_Reduction
cd WellCo_Churn_Reduction
```

3. Install dependencies
   
`pip install -r requirements.txt`

5. Provide dataset path (optional) - Since the data is sensitive and can't be publish

If the official dataset is available, place the CSV files inside:

data/train/
data/test/

And write the path to the data folder when asked while running the code

If no dataset is provided, the pipeline automatically generates synthetic data so the full pipeline remains reproducible.

4. Run the pipeline
   
`python main.py`

The script will ask for a dataset path. If the path is not provided or files are missing, synthetic data will be used automatically.

## Outputs

After execution, the following outputs are generated:

- outputs/outreach_list.csv — ranked list of members for outreach

- outputs/s_learner_model.pkl — trained model

- outputs/figures/ — performance and uplift visualizations

- outputs/feature_names.txt — feature metadata

Example pipeline results:

- Training AUC: 0.8119

- Validation AUC: 0.6779

- Estimated optimal outreach size: 3,903 members

  
## Methodology
### Feature Engineering and Selection

Feature engineering was performed using multiple behavioral and clinical data sources to capture member engagement and health-risk indicators based on the compant prefrence:

### App Usage Features: 

- Total number of session 
  
- Total number of days the member was active in the app

- Average sessions per day

- Precentage of days the member was active in the app

- Max sessions in single day

- Number of days since last session

- Sessions in the last 7 days

- Week2 app usage vs week1 app usage

### Web Visit Features: proportion of health-related content consumed

- Number of total web visits

- Number of days the member was active in the web

- Precantage of health content from all web visits

- Precentage of visit in WellCo website from all health content visits

- If visited on contant about diabetes

- If visited on contant about hypertension

- If visites on contant about nutrition

- Number of days since last web visit

- Number of web visits in the last 7 days

- Number of web visits week2 compare to week1

### Claims Features: indicators for chronic or priority medical conditions

- Total number of claims

- Number of distinct diagnoses

- If has diabetes

- If has hypertension

- If need dietary counsel

- Has one of the 3 priority condition

- Number of priority conditions

- Numbers of days since recent diagnosis

- Did he has recent diagnos

- Number of days since first diagnosis

### Member Features: tenure (days since signup), historical outreach exposure

- Number of days since signup

- Is new member (join in the last 30 days)

- Is mature member (a member more then a year)

### Feature from combineing the files

- Has diabetes but did not look at diabetes content

- Has hypertensionbut did not look at hypertension content

- Has dietary counsel but did not look at nutrition content

- Have priority condition but look at low health content

- Is app and web declining

- Content alignment with condition score

- Sessions in app per active day in website

- Is recent engagement in the app

Feature selection was guided by:

Domain relevance: engagement and health status are strong churn drivers, based on the WellCo client brief - focusing on 3 diagnose, and health contant such as exrecise, sleep, stress. 

Data quality: features with high completeness and reliability were prioritized

Predictive contribution: features improving validation AUC were retained

The final training matrix contained 39 features across 10,000 members.

## Model Approach

An uplift modeling S-Learner framework was implemented using XGBoost:

- Model predicts churn probability conditional on outreach exposure

- Treatment effect (CATE) - Difference in predicted churn risk with outreach vs. without outreach, is computed as: P(churn | no outreach) − P(churn | outreach)

- Calculate the weighted uplift - cate * probability to churn without treatment * WellCo preferance (if have priority diagnose, health content, if is mature member) 

- Members are ranked by weighted upliftt and the optimal n was chosen using the elbow method.

This approach prioritizes members most likely to benefit from outreach, not only those with high churn risk.

## Model Evaluation

### The S learner Model performance was evaluated using a train-validation split and standard classification metrics.

Performance results:

Train ROC-AUC: 0.8119

Validation ROC-AUC: 0.6779

Overfitting gap: 0.1340

Validation classification metrics:

Metric	  Non-Churn	  Churn

Precision	  0.87	  0.32

Recall	  0.69	  0.58

F1-score	  0.77	  0.41

ROC-AUC was selected as the primary metric because:

- The dataset is moderately imbalanced (~20% churn)

- Ranking quality is critical for outreach prioritization

- Outreach decisioning depends on probability ordering rather than strict classification thresholds

### Using Outreach Data in Modelling

The dataset contains a historical outreach event occurring between the observation window and the churn outcome period.
This information was incorporated as the treatment indicator in an S-Learner uplift modeling framework:

- The model predicts churn probability conditional on outreach exposure.

- Conditional Average Treatment Effect (CATE) is estimated as the difference between predicted churn risk with outreach and without outreach.

- This enables prioritization of members who are most likely to benefit from outreach, rather than those who are simply high-risk.

            Training and testing results:

Mean CATE: 0.0417    0.0414

Median CATE: 0.0344    0.0345

% positive treatment effect: 96.9%    96.8%

This enables decision-making based on incremental retention impact, rather than raw churn prediction alone.

## Selecting n (Outreach Size)

Because outreach cost is constant but **unknown**, the optimal outreach size was determined using incremental value analysis:

Members were ranked by weighted uplift.

The elbow point of the curve was used to determine the optimal outreach size.

In the final test application, the optimal outreach size was:

n = 3,903 members

This approach balances expected retention gains against diminishing marginal benefit as additional lower-impact members are included.
