import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def remove_outliers_iqr(df, feature, factor=1.5):
    """Remove outliers based on interquartile range"""
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

def calc_pct_diff(row):
    """Calculate percentage difference between latest and earliest bill amount."""
    # Use the first non-zero value in BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    left_val = 0
    for col in bill_cols:
        if row[col] != 0 and row[col] > 0:
            left_val = row[col]
            break # Once the first nonzero value is found
    
    # Use the first non-zero value in BILL_AMT6, BILL_AMT5, BILL_AMT4, BILL_AMT3, BILL_AMT2, BILL_AMT1
    right_val = 0
    for col in bill_cols[::-1]:
        if row[col] != 0 and row[col] > 0:
            right_val = row[col]
            break
        
    if right_val == 0 and left_val == 0:
        return 0
    return (left_val / right_val - 1)* 100

def preprocess_data(path):
    """Clean, and preprocess the credit card dataset."""
    # Load the dataset
    df_default_raw = pd.read_csv(f"{path}\\UCI_Credit_Card.csv", index_col=0)
    
    # Data Cleaning
    categorical_vars = ["SEX", "EDUCATION", "MARRIAGE"]
    df_default_raw[categorical_vars] = df_default_raw[categorical_vars].astype("category")
    
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    
    df_default_clean = remove_outliers_iqr(df_default_raw, "LIMIT_BAL")
    
    # Feature Engineering
    df_default_clean["AVG_BILL_AMT"] = df_default_clean[bill_cols].mean(axis=1)
    df_default_clean["AVG_PAY_AMT"] = df_default_clean[pay_amt_cols].mean(axis=1)
    df_default_clean["TOTAL_BILL_AMT"] = df_default_clean[bill_cols].sum(axis=1)
    df_default_clean["CREDIT_UTILIZATION"] = df_default_clean["TOTAL_BILL_AMT"] / (df_default_clean["LIMIT_BAL"] + 1)
    df_default_clean["AVG_MONTHLY_UTILIZATION"] = df_default_clean[bill_cols].div(df_default_clean["LIMIT_BAL"], axis=0).mean(axis=1)
    
    # Maximum delay on payment
    delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df_default_clean["MAX_DELAY"] = (df_default_clean[delay_cols] > 0).sum(axis=1)
    
    # Bill trend
    df_default_clean["BILL_TREND"] = df_default_clean.apply(calc_pct_diff, axis=1)
    
    # Interaction feature
    df_default_clean["AGE_LIMIT_INTERACTION"] = df_default_clean["AGE"] * df_default_clean["LIMIT_BAL"]
    
    # Drop ID
    if "ID" in df_default_clean.columns:
        df_default_clean.drop("ID", axis=1, inplace=True)
    
    # Encoding categorical variables
    ct = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(drop="first", sparse_output=False), categorical_vars)
        ],
        remainder="passthrough"
    )
    X = df_default_clean.drop("default.payment.next.month", axis=1)
    X_transformed = ct.fit_transform(X)
    
    # Create a df from the transformed data
    new_columns = [col.split("__")[1] for col in ct.get_feature_names_out()]
    df_X_transformed = pd.DataFrame(X_transformed, columns=new_columns)
    
    # Scaling numerical features
    labels_to_drop = categorical_vars + ["default.payment.next.month"]
    numerical_features = [col for col in df_default_clean.columns if col not in labels_to_drop]
    scaler = StandardScaler()
    df_X_transformed[numerical_features] = scaler.fit_transform(df_X_transformed[numerical_features])
    
    Y = df_default_clean["default.payment.next.month"]
    default_data = {"Raw": df_default_raw, "Cleaned": df_default_clean}
    
    return default_data, df_X_transformed, Y