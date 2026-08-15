import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

np.random.seed(42)

def generate_mock_data(n_samples=2000):
    is_fake = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    followers = np.where(is_fake == 1, np.random.randint(0, 200, n_samples), np.random.randint(50, 10000, n_samples))
    following = np.where(is_fake == 1, np.random.randint(500, 5000, n_samples), np.random.randint(50, 2000, n_samples))
    posts_count = np.where(is_fake == 1, np.random.randint(0, 15, n_samples), np.random.randint(10, 800, n_samples))
    has_profile_pic = np.where(is_fake == 1, np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples), np.random.choice([0, 1], p=[0.05, 0.95], size=n_samples))
    bio_length = np.where(is_fake == 1, np.random.randint(0, 20, n_samples), np.random.randint(15, 150, n_samples))
    has_external_url = np.where(is_fake == 1, np.random.choice([0, 1], p=[0.6, 0.4], size=n_samples), np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples))
    account_age_days = np.where(is_fake == 1, np.random.randint(1, 90, n_samples), np.random.randint(100, 3000, n_samples))
    username_digits = np.where(is_fake == 1, np.random.randint(3, 8, n_samples), np.random.randint(0, 3, n_samples))
    username_len = np.random.randint(6, 15, n_samples)
    
    return pd.DataFrame({
        'username_len': username_len,
        'username_digits': username_digits,
        'has_profile_pic': has_profile_pic,
        'bio_length': bio_length,
        'has_external_url': has_external_url,
        'followers': followers,
        'following': following,
        'posts_count': posts_count,
        'account_age_days': account_age_days,
        'is_fake': is_fake
    })

def engineer_features(df):
    df_feat = df.copy()
    df_feat['digit_to_length_ratio'] = df_feat['username_digits'] / (df_feat['username_len'] + 1e-5)
    df_feat['follower_following_ratio'] = df_feat['followers'] / (df_feat['following'] + 1)
    df_feat['network_imbalance'] = df_feat['following'] - df_feat['followers']
    df_feat['posting_rate_per_day'] = df_feat['posts_count'] / (df_feat['account_age_days'] + 1)
    df_feat['profile_completeness'] = (
        df_feat['has_profile_pic'] + 
        (df_feat['bio_length'] > 0).astype(int) + 
        df_feat['has_external_url']
    )
    df_feat['is_mass_follower'] = ((df_feat['following'] > 1000) & (df_feat['followers'] < 100)).astype(int)
    return df_feat

if __name__ == "__main__":
    raw_df = generate_mock_data()
    df = engineer_features(raw_df)
    
    X = df.drop(columns=['is_fake'])
    y = df['is_fake']
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)
    
    os.makedirs('model', exist_ok=True)
    model.save_model('model/fake_account_xgb.json')
    print("Model trained and saved to model/fake_account_xgb.json!")