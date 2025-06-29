import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from utils import get_feature_columns, get_target_columns, preprocess_data

# Load dataset
df = pd.read_csv("data/warehouse_dataset_polished.csv")

# Filter training data (2023 + 2024)
train_df = df[df["year"].isin([2023, 2024])]

# Setup directories
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

features = get_feature_columns()
targets = get_target_columns()

# Preprocess input features
X, preprocessor = preprocess_data(train_df[features])

for product in targets:
    y = train_df[product]
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    joblib.dump(model, f"models/{product}_model.pkl")
    print(f"✅ Trained and saved: {product}_model.pkl")
