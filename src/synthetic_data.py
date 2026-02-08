
# Synthetic fallback generator
import pandas as pd
import numpy as np
import os

def generate_synthetic_data(save_path=None, is_test = False, num_members=1000):
    print("Generating synthetic data...")

    app_usage = pd.DataFrame({
        'member_id': np.random.randint(1, num_members+1, size=num_members*5),
        'timestamp': pd.date_range("2025-01-01", periods=num_members*5, freq='D')
    })

    if is_test == False:
        churn_labels = pd.DataFrame({
            'member_id': range(1, num_members+1),
            'signup_date': pd.date_range("2024-01-01", periods=num_members, freq='D'),
            'churn': np.random.randint(0, 2, size=num_members),
            'outreach': np.random.randint(0, 2, size=num_members)
        })
    else:
        churn_labels = pd.DataFrame({
            'member_id': range(1, num_members+1),
            'signup_date': pd.date_range("2024-01-01", periods=num_members, freq='D')
        })

    claims = pd.DataFrame({
        'member_id': np.random.randint(1, num_members+1, size=num_members*2),
        'diagnosis_date': pd.date_range("2024-06-01", periods=num_members*2, freq='D'),
        'icd_code': np.random.choice(['Z71.3', 'J00', 'M54.5', 'I10', 'E11.9', 'K21.9', 'R51', 'A09',
       'B34.9', 'H10.9'], size=num_members*2)
    })

    # Generate random URLs for synthetic web_visits
    base_domains = [
        'https://health.wellco/',
        'https://care.portal/',
        'https://guide.wellness/',
        'https://living.better/',
        'https://portal.site/',
        'https://example.com/',
        'https://world.news/',
        'https://media.hub/'
    ]

    paths = [
        'chronic', 'heart', 'diabetes', 'strength', 'sleep', 'nutrition', 'mindfulness',
        'cardio', 'hypertension', 'fitness', 'aerobic', 'stress', 'weight', 'recipes',
        'wellness', 'cars', 'sports', 'travel', 'tech', 'movies', 'gaming', 'pets', 'finance'
    ]

    titles = ['Diabetes management', 'Gadget roundup', 'Hypertension basics',
       'Game reviews', 'Stress reduction', 'Restorative sleep tips',
       'Healthy eating guide', 'Aerobic exercise', 'HbA1c targets',
       'Strength training basics', 'Lowering blood pressure',
       'New releases', 'Sleep hygiene', 'Cardio workouts',
       'Mediterranean diet', 'Match highlights', 'Exercise routines',
       'Meditation guide', 'Dog training', 'Cardiometabolic health',
       'Electric vehicles', 'Budget planning', 'Top destinations',
       'High-fiber meals', 'Cholesterol friendly foods',
       'Weight management']
    
    descriptions = ['Blood sugar and glycemic control', 'Smartphones and laptops news',
       'Blood pressure and lifestyle changes',
       'Strategy tips and updates', 'Mindfulness and wellness',
       'Sleep apnea screening and hygiene',
       'Tips on nutrition and balanced diets',
       'Cardiovascular fitness and endurance',
       'Improving glycemic control and blood glucose',
       'Muscle building and metabolism',
       'Lifestyle changes and medication adherence',
       'Box office and trailers',
       'Improve sleep quality and reduce stress', 'Exercise and recovery',
       'Nutrition patterns and heart health',
       'League standings and transfers', 'Cardio and strength workouts',
       'Stress management and mindfulness',
       'Obedience and behavior basics',
       'Diet, exercise, and risk factors', 'Charging networks and range',
       'Household expenses and savings', 'City guides and itineraries',
       'Balanced nutrition and glycemic control',
       'Lowering LDL and improving lipid profile',
       'Sustainable healthy eating']
    
    # Build random URLs
    urls = [np.random.choice(base_domains) + np.random.choice(paths) for _ in range(num_members*3)]

    web_visits = pd.DataFrame({
        'member_id': np.random.randint(1, num_members+1, size=num_members*3),
        'timestamp': pd.date_range("2025-01-01", periods=num_members*3, freq='D'),
        'url': urls,
        'title': np.random.choice(titles),
        'description': np.random.choice(descriptions)
    })

    # Save CSVs if save_path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        app_usage.to_csv(os.path.join(save_path, "app_usage.csv"), index=False)
        churn_labels.to_csv(os.path.join(save_path, "churn_labels.csv"), index=False)
        claims.to_csv(os.path.join(save_path, "claims.csv"), index=False)
        web_visits.to_csv(os.path.join(save_path, "web_visits.csv"), index=False)
        print(f"Synthetic CSVs saved to folder: {save_path}")

    return app_usage, churn_labels, claims, web_visits
