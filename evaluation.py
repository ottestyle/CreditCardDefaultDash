import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Function to evaluate models, extract metrics and feature importance"""
    # Fitting the model using the training data
    model_choice = model
    model_choice.fit(X_train, y_train)    
    
    # Predictions and probabilities
    y_pred = model_choice.predict(X_test)
    y_proba = model_choice.predict_proba(X_test)[:, 1]
    
    # Classification report
    c_report = classification_report(y_test, y_pred)
    
    # Confusion matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC AUC Score
    auc_score = roc_auc_score(y_test, y_proba)
    
    # Features
    if model_name == "Logistic Regression":
        df_features = pd.DataFrame({
            "Feature": X_train.columns,
            "LogReg Coefficient": model_choice.coef_[0],
            "Abs Coefficient": np.abs(model_choice.coef_[0]),
            "Odds Ratio": np.exp(model_choice.coef_[0])
        })
    else:
        df_features = pd.DataFrame({
            "Feature": X_train.columns,
            "RF Importance": model_choice.feature_importances_
        })
    
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classification_report": c_report,
        "confusion_matrix": c_matrix,
        "auc_score": auc_score, 
        "feature_importance": df_features
    }