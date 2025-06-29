import pandas as pd
import joblib
import os

from utils import get_feature_columns, get_target_columns, preprocess_data

# # Load dataset
# df = pd.read_csv("data/warehouse_2025_template.csv")

# # Filter test data (2025)
# test_df = df[df["year"] == 2025]

test_df = pd.read_csv("data/warehouse_2025_template.csv")

# Load feature columns
features = get_feature_columns()
targets = get_target_columns()

# Preprocess test input
X, _ = preprocess_data(test_df[features])  # just get the transformer object again

# Prepare result DataFrame
results = test_df[["month_number", "year", "warehouse_location"]].copy()

for product in targets:
    model_path = f"models/{product}_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        results[product] = model.predict(X)
    else:
        print(f"⚠️ Model not found: {model_path}")

# Save predictions
results.to_csv("output/predictions_2025.csv", index=False)
print("✅ All predictions saved to output/predictions_2025.csv")
