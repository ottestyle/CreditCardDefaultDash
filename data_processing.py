import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler

# def remove_outliers_iqr(df, feature, factor=1.5):
#     """Remove outliers based on interquartile range"""
#     Q1 = df[feature].quantile(0.25)
#     Q3 = df[feature].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - factor * IQR
#     upper_bound = Q3 + factor * IQR
#     return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

def remove_outliers_percentile(df, feature, lower_pct=0.005, upper_pct=0.995):
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

def preprocess_data(path):
    """Clean, and preprocess the credit card dataset."""
    # Load the dataset
    df_default_raw = pd.read_csv(f"{path}\\UCI_Credit_Card.csv", index_col=0)
    
    # Data Cleaning
    df_default_raw.columns = [col.replace("_", " ").lower().title() for col in df_default_raw.columns]
    df_default_raw = df_default_raw.rename(columns={"Pay 0": "Pay0", "Pay 2": "Pay2", "Pay 3": "Pay3", "Pay 4": "Pay4", "Pay 5": "Pay5", "Pay 6": "Pay6","Default.Payment.Next.Month": "Default Payment Next Month"})
    
    df_default_clean = df_default_raw
    
    bill_cols = [f"Bill Amt{i}" for i in range(1,7)]
    pay_amt_cols = [f"Pay Amt{i}" for i in range(1,7)]
    categorical_vars = ["Sex", "Education", "Marriage", "Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6"]
    
    # Feature Engineering
    df_default_clean["Avg Bill Amt"] = df_default_clean[bill_cols].mean(axis=1)
    df_default_clean["Avg Pay Amt"] = df_default_clean[pay_amt_cols].mean(axis=1)
    df_default_clean["Total Bill Amt"] = df_default_clean[bill_cols].sum(axis=1)
    df_default_clean["Credit Utilization"] = df_default_clean["Total Bill Amt"] / (df_default_clean["Limit Bal"] + 1)
    df_default_clean["Avg Monthly Utilization"] = df_default_clean[bill_cols].div(df_default_clean["Limit Bal"], axis=0).mean(axis=1)
    
    # Maximum delay on payment
    delay_cols = ["Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6"]
    df_default_clean["Max Delay"] = (df_default_clean[delay_cols] > 0).sum(axis=1)
    
    # Bill- and Pay trend
    df_default_clean["Bill Trend"] = df_default_clean.apply(lambda row: calc_trend("Bill", row), axis=1)
    df_default_clean["Pay Trend"] = df_default_clean.apply(lambda row: calc_trend("Pay", row), axis=1)
    
    # Interaction feature
    df_default_clean["Age Limit Interaction"] = df_default_clean["Age"] * df_default_clean["Limit Bal"]
    
    # Remove outliers in bill cols
    for col in bill_cols:
        df_default_clean = remove_outliers_percentile(df_default_clean, col)
        
    # Capping bill and pay trend
    for trend in ["Bill Trend", "Pay Trend"]:
        lower_cap = df_default_clean[trend].quantile(0.01)
        upper_cap = df_default_clean[trend].quantile(0.99)
        df_default_clean[trend] = df_default_clean[trend].clip(lower=lower_cap, upper=upper_cap)
        
    # Yeo-Johnson transformation
    pt = PowerTransformer(method="yeo-johnson")
    financial_cols = bill_cols + pay_amt_cols + ["Avg Bill Amt", "Avg Pay Amt", "Total Bill Amt"]
    df_default_clean[financial_cols] = pt.fit_transform(df_default_clean[financial_cols])
     
    # Drop ID
    if "ID" in df_default_clean.columns:
        df_default_clean.drop("ID", axis=1, inplace=True)
    
    df_default_clean[categorical_vars] = df_default_clean[categorical_vars].astype("category")
    
    # Encoding categorical variables
    ct = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=False), categorical_vars)
        ],
        remainder="passthrough"
    )
    X = df_default_clean.drop("Default Payment Next Month", axis=1)
    X_transformed = ct.fit_transform(X)
    
    # Create a df from the transformed data
    new_columns = [col.split("__")[1] for col in ct.get_feature_names_out()]
    df_X_transformed = pd.DataFrame(X_transformed, columns=new_columns)
    
    # Scaling numerical features
    labels_to_drop = categorical_vars + ["Default Payment Next Month"]
    numerical_features = [col for col in df_default_clean.columns if col not in labels_to_drop]
    scaler = StandardScaler()
    df_X_transformed[numerical_features] = scaler.fit_transform(df_X_transformed[numerical_features])
    
    Y = df_default_clean["Default Payment Next Month"]
    default_data = {"Raw": df_default_raw, "Cleaned": df_default_clean}
    
    return default_data, df_X_transformed, Y