# data_processing.py
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler

def remove_outliers_percentile(df, feature, lower_pct=0.01, upper_pct=0.99):
    """Remove observations outside the given percentile range"""
    lower_val = df[feature].quantile(lower_pct)
    upper_val = df[feature].quantile(upper_pct)
    return df[(df[feature] >= lower_val) & (df[feature] <= upper_val)]

def calc_trend(feature, row):
    """Calculate bill- and pay trend as the slope of bill/pay amounts over time via linear regression."""
    # Use the first non-zero value in Bill/Pay Amt1, ... , Bill/Pay Amt6
    cols = [f"{feature} Amt{i}" for i in range(1,7)]
    y = row[cols].values.astype(float)
    x = np.arange(len(cols), 0, -1) #[6, ..., 1]
    
    # Use only non-zero values
    mask = y > 0
    if np.sum(mask) < 2:
        return 0
    x = x[mask]
    y = y[mask]
    
    slope, intercept = np.polyfit(x, y, 1)
    return slope

def load_and_engineer_data(path):
    """Load raw data, clean it, and perform feature engineering."""
    df_raw = pd.read_csv(f"{path}\\UCI_Credit_Card.csv", index_col=0)
    
    # Clean column names
    df_raw.columns = [col.replace("_", " ").lower().title() for col in df_raw.columns]
    df_raw = df_raw.rename(columns={
        "Pay 0": "Pay0", "Pay 2": "Pay2", "Pay 3": "Pay3", 
        "Pay 4": "Pay4", "Pay 5": "Pay5", "Pay 6": "Pay6", 
        "Default.Payment.Next.Month": "Default Payment Next Month"
        })
    
    df = df_raw.copy()
    
    bill_cols = [f"Bill Amt{i}" for i in range(1,7)]
    pay_amt_cols = [f"Pay Amt{i}" for i in range(1,7)]
    categorical_vars = ["Sex", "Education", "Marriage", "Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6", "Max Delay"]
    delay_cols = ["Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6"]
    log_cols = ["Limit Bal", "Credit Utilization"]
    
    # Remove outliers in bill cols
    for col in bill_cols:
        df = remove_outliers_percentile(df, col)
    
    # Feature Engineering
    df["Avg Bill Amt"] = df[bill_cols].mean(axis=1)
    df["Avg Pay Amt"] = df[pay_amt_cols].mean(axis=1)
    df["Total Bill Amt"] = df[bill_cols].sum(axis=1)
    df["Credit Utilization"] = df["Total Bill Amt"] / (df["Limit Bal"] + 1)
    df["Avg Monthly Utilization"] = df[bill_cols].div(df["Limit Bal"], axis=0).mean(axis=1)
    
    # Maximum delay on payment
    df["Max Delay"] = (df[delay_cols] > 0).sum(axis=1)
    
    # Bill- and Pay trend
    df["Bill Trend"] = df.apply(lambda row: calc_trend("Bill", row), axis=1)
    df["Pay Trend"] = df.apply(lambda row: calc_trend("Pay", row), axis=1)
    
    # Interaction feature
    df["Age Limit Interaction"] = df["Age"] * df["Limit Bal"]
        
    # Capping bill and pay trend
    for trend in ["Bill Trend", "Pay Trend"]:
        lower_cap = df[trend].quantile(0.01)
        upper_cap = df[trend].quantile(0.99)
        df[trend] = df[trend].clip(lower=lower_cap, upper=upper_cap)
        
    # Multicollinearity Analysis (VIF)
    features_for_vif = [
        "Limit Bal", "Age", "Avg Bill Amt", "Avg Pay Amt", "Total Bill Amt",
        "Credit Utilization", "Avg Monthly Utilization", "Bill Trend", "Pay Trend",
        "Age Limit Interaction", "Max Delay",
        "Bill Amt1", "Bill Amt2", "Bill Amt3", "Bill Amt4", "Bill Amt5", "Bill Amt6",
        "Pay Amt1", "Pay Amt2", "Pay Amt3", "Pay Amt4", "Pay Amt5", "Pay Amt6",
        "Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6",
        "Sex", "Education", "Marriage"
        ]
    
    X_vif = df[features_for_vif]
    
    # Standardize features to ensure comparability in VIF calculations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vif)
        
    # VIF first results
    df_vif = pd.DataFrame(X_vif.columns, columns=["Feature"])
    df_vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(len(X_vif.columns))]
    
    # Addressing the interaction collinearity by centering
    df["Age Limit Interaction"] = (df["Age"] - df["Age"].mean()) * (df["Limit Bal"] - df["Limit Bal"].mean())
    
    columns_to_drop = [
        "Default Payment Next Month", "Avg Bill Amt", "Avg Pay Amt", 
        "Bill Amt1", "Bill Amt2", "Bill Amt3", "Bill Amt4", "Bill Amt5", "Bill Amt6",
        "Pay Amt1", "Pay Amt2", "Pay Amt3", "Pay Amt4", "Pay Amt5", "Pay Amt6",
        "Avg Monthly Utilization"
    ]
    
    X_vif_reduced = df.drop(columns=columns_to_drop, errors="ignore")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vif_reduced)
    
    # VIF second results
    df_vif = pd.DataFrame(X_vif_reduced.columns, columns=["Feature"])
    df_vif["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(len(X_vif_reduced.columns))]
    
    # Removing redudant features
    df = df.drop(columns=columns_to_drop[1:])
    
    # Log transformation of right skewed distributions
    for col in log_cols:
        df[col] = np.log(df[col] + 1)
    
    return df_raw, df, categorical_vars, df_vif

def build_preprocessing_pipeline(categorical_vars):
    """
    Preprocessing pipeline to transform financial variables and encode categoricals.
    Only fitted to the training data
    """
    other_numeric_vars = ["Limit Bal", "Age", "Credit Utilization", "Bill Trend", "Pay Trend", "Age Limit Interaction"]
    ordinal_vars = ["Education", "Max Delay"]
    nominal_vars = [var for var in categorical_vars if var not in ordinal_vars]
    
    ct = ColumnTransformer(
        transformers=[
            ("financial", PowerTransformer(method="yeo-johnson"), ["Total Bill Amt"]),
            ("numeric", StandardScaler(), other_numeric_vars),
            ("ordinal", OrdinalEncoder(), ordinal_vars),
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_vars)
            ],
        remainder="passthrough" 
        )
    return ct

def preprocess_data(path):
    """Load data, feature engineer, and return both raw and processed data"""
    df_raw, df_cleaned, categorical_vars, df_vif = load_and_engineer_data(path)
    
    # Separate target from features
    X = df_cleaned.drop("Default Payment Next Month", axis=1)
    Y = df_cleaned["Default Payment Next Month"]
    
    default_data = {"Raw": df_raw, "Cleaned": df_cleaned}
    
    return default_data, X, Y, categorical_vars, df_vif