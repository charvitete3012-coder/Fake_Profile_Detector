from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import os

app = FastAPI(title="Fake Social Media Account Detector API")

# Load model on start
model_path = os.path.join("model", "fake_account_xgb.json")
model = xgb.XGBClassifier()
model.load_model(model_path)

class ProfileData(BaseModel):
    username_len: int
    username_digits: int
    has_profile_pic: int
    bio_length: int
    has_external_url: int
    followers: int
    following: int
    posts_count: int
    account_age_days: int

def engineer_features(data: dict):
    df = pd.DataFrame([data])
    df['digit_to_length_ratio'] = df['username_digits'] / (df['username_len'] + 1e-5)
    df['follower_following_ratio'] = df['followers'] / (df['following'] + 1)
    df['network_imbalance'] = df['following'] - df['followers']
    df['posting_rate_per_day'] = df['posts_count'] / (df['account_age_days'] + 1)
    df['profile_completeness'] = (
        df['has_profile_pic'] + 
        (df['bio_length'] > 0).astype(int) + 
        df['has_external_url']
    )
    df['is_mass_follower'] = ((df['following'] > 1000) & (df['followers'] < 100)).astype(int)
    return df

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict_account(profile: ProfileData):
    raw_input = profile.model_dump()
    X = engineer_features(raw_input)
    
    prob_fake = float(model.predict_proba(X)[0][1])
    is_fake = bool(prob_fake >= 0.5)
    
    # Calculate key flags for explainability
    flags = []
    if X['digit_to_length_ratio'].iloc[0] > 0.3:
        flags.append("High ratio of digits in username")
    if X['follower_following_ratio'].iloc[0] < 0.1:
        flags.append("Extremely high following-to-follower ratio")
    if raw_input['has_profile_pic'] == 0:
        flags.append("Missing profile picture")
    if raw_input['account_age_days'] < 30 and raw_input['following'] > 500:
        flags.append("Suspiciously rapid following pattern on new account")
        
    return {
        "is_fake": is_fake,
        "fake_probability": round(prob_fake * 100, 2),
        "risk_level": "HIGH" if prob_fake > 0.7 else ("MEDIUM" if prob_fake > 0.4 else "LOW"),
        "detected_flags": flags
    }