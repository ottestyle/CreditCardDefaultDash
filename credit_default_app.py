# credit_default_app.py
import os
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import pandas as pd

from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc

# Load data
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# Change directory to app, modules and css 
os.chdir(os.environ["DEFAULT_OF_CREDIT_CARD_CLIENTS"])

from data_processing import preprocess_data, build_preprocessing_pipeline
from evaluation import evaluate_model
from layouts.data_layout import data_layout
from layouts.models_layout import models_layout
from layouts.model_comparison_layout import model_comparison_layout
from callbacks import register_callbacks

# Preprocess data: raw vs. cleaned, features and target
default_data, X, Y, categorical_vars, df_vif = preprocess_data(path)

# Train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Build and fit the transformation pipeline on the training set
pipeline = build_preprocessing_pipeline(categorical_vars)
X_train_transformed = pipeline.fit_transform(X_train)

# Use the fitted pipeline to transform the test set
X_test_transformed = pipeline.transform(X_test)

# Convert back to df
feature_names = [col.split("__")[1] for col in pipeline.get_feature_names_out()]
X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
X_train_transformed = X_train_transformed.apply(pd.to_numeric)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
X_test_transformed = X_test_transformed.apply(pd.to_numeric)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), # Baseline model
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
}

# Evaluate each model
eval_dict = {}
for model_name, model in models.items():
    eval_dict[model_name] = evaluate_model(model, X_train_transformed, X_test_transformed, y_train, y_test, model_name)

############
# Dash App #
############

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY], assets_folder="assets")

######################
# Overall App Layout #
######################
app.layout = dbc.Container(
    fluid=True,
    children=[
        # Keeps track of the URL so it can display different content
        dcc.Location(id="url", refresh=False),
        
        # Header row with H1 in the top left, gray background and white text
        dbc.Row(
            dbc.Col(
                html.H1("Default of Credit Card Clients in Taiwan",
                        className="bg-primary text-white",
                        style={
                            "padding": "10px",
                            "textAlign": "left"
                            }
                        ),
                    width=12
                    )
            ),
        
        # Navigation links row with active state styling -> custom.css
        dbc.Row(
            dbc.Col(
                dbc.Nav(
                    [
                        dbc.NavLink("Data", 
                                    href="/data", 
                                    active="exact", 
                                    style={"color": "black"}
                                    ),
                        dbc.NavLink("Models", 
                                    href="/models", 
                                    active="exact", 
                                    style={"color": "black"}
                                    ),
                        dbc.NavLink("Model Comparison", 
                                    href="/model-comparison", 
                                    active="exact", 
                                    style={"color": "black"}
                                    ),
                    ],
                    pills=True,  # Rounded corners
                    className="mb-4", # Spacing
                ),
                width=12
            )
        ),
        
        # Content area for displaying page-specific content
        dbc.Row(
            dbc.Col(
                html.Div(id="page-content")
            )
        )
    ])
    
##############################################################
# Callback to control page layout rendering based on the url #
##############################################################
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
    )
def display_page(pathname):
    """
    This callback function returns the content for the page based on the URL pathname.
    - If the pathname is '/' it will be redirected to the data content.
    - If the pathname is '/data' the data content is shown.
    - If the pathname is '/models' results of the models are shown.
    - If the pathname is '/model-comparison', comparison of models is shown.
    - Otherwise, a 404 error message is displayed.
    """
    if pathname == "/":
        return dcc.Location(pathname="/data", id="redirect")
    elif pathname == "/data":
        return data_layout(default_data)
    elif pathname == "/models":
        return models_layout(models)
    elif pathname == "/model-comparison":
        return model_comparison_layout(eval_dict)
    else:
        return html.Div([
            html.H3("404: Page not found", style={"textAlign": "center"})])
    
# Register all callbacks
register_callbacks(app, default_data, df_vif, eval_dict, y_test)

# Run the app    
if __name__ == '__main__': #http://127.0.0.1:8050/
    app.run_server(debug=True)