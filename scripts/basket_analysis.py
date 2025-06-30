import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def get_target_columns():
    return [
        "maggi", "desi_ghee", "dairy_milk", "lays", "coke", "amul_butter",
        "kurkure", "frooti", "parle_g", "nescafe", "pepsi", "oreo", "kitkat", "maaza",
        "good_day", "real_juice", "thumbs_up", "sprite", "bourbon", "hide_and_seek",
        "little_debbie", "amul_milk", "amul_cheese", "bournvita", "horlicks",
        "detergent_powder", "surf_excel", "vim_bar", "colgate", "toothbrush",
        "soap", "shampoo", "conditioner", "toilet_cleaner", "phenyl", "harpic",
        "detergent_liquid", "salt", "sugar", "rice", "wheat_flour",
        "pulses", "rajma", "chana", "soya_chunks", "bread", "butter", "jam", "pickles", "paneer"
    ]

# Load data
df = pd.read_csv("data/warehouse_dataset_polished.csv")
targets = get_target_columns()

# Focus on top 25 products to reduce memory usage
top_items = df[targets].sum().sort_values(ascending=False).head(25).index.tolist()
basket_df = (df[top_items].fillna(0) > 0).astype(int)

# Sample 500 rows max
if len(basket_df) > 500:
    basket_df = basket_df.sample(500, random_state=42)

# Run FP-Growth
frequent_itemsets = fpgrowth(basket_df, min_support=0.02, use_colnames=True, max_len=2)

# Generate less strict rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[rules["lift"] > 1.0]

# Sort and save if not empty
if not rules.empty:
    rules = rules.sort_values("confidence", ascending=False)
    rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_csv("output/cabinet_suggestions.csv", index=False)
    print("✅ Rules generated and saved.")
else:
    print("⚠️ No valid rules found. Try lowering thresholds or increasing data.")
