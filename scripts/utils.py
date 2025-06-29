from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def get_feature_columns():
    return [
        "month_number", "warehouse_location", "event", "avg_price", "avg_discount",
        "day_count_event", "weather_trend", "promotion", "stock_out_flag"
    ]

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

def preprocess_data(df):
    categorical = ["warehouse_location", "event", "weather_trend"]
    numerical = ["month_number", "avg_price", "avg_discount", "day_count_event", "promotion", "stock_out_flag"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="mean"), numerical)
    ])
    return df, preprocessor
