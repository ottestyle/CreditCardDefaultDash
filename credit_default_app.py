import os
import kagglehub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from dash import Dash, html, Input, Output, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px

def target_bar_plot(df, data_state):
    count_target_data = df[data_state]["Default.Payment.Next.Month"].value_counts().sort_index().reset_index()

    return px.bar(count_target_data,
                  x="Default.Payment.Next.Month",
                  y="count",
                  labels={"Default.Payment.Next.Month": "Default (0 = No, 1 = Yes)",
                          "count": "Count"})
     
pio.renderers.default="browser"

# Load data
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# Change directory to app, modules and css 
os.chdir(os.environ["DEFAULT_OF_CREDIT_CARD_CLIENTS"])

from data_processing import preprocess_data
from evaluation import evaluate_model

eval_dict = {}

# Preprocess data
default_data, X, Y = preprocess_data(path)

for key, df in default_data.items():
    df.columns = [col.lower().title() for col in df.columns]

# Train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42), # Baseline model
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500)
}

for model_name, model in models.items():
    eval_dict[model_name] = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)

# List with continuous features
continuous_var = ["Limit_Bal", "Age", "Bill_Amt1", "Bill_Amt2", "Bill_Amt3", "Bill_Amt4", "Bill_Amt5", "Bill_Amt6", "Pay_Amt1",
                  "Pay_Amt2", "Pay_Amt3", "Pay_Amt4", "Pay_Amt5", "Pay_Amt6", "Total_Bill_Amt", "Credit_Utilization",
                  "Avg_Monthly_Utilization", "Bill_Trend", "Age_Limit_Interaction"]

# Static plots for distribution of target as well target vs. credit limit
count_target_data = default_data["Cleaned"]["Default.Payment.Next.Month"].value_counts().sort_index().reset_index()

fig_target_bar = px.bar(count_target_data, 
                        x="Default.Payment.Next.Month",
                        y="count",
                        labels={"Default.Payment.Next.Month": "Default (0 = No, 1 = Yes)",
                                "count": "Count"})
fig_target_bar_title = "Default Payment Next Month"

fig_target_box = px.box(default_data["Cleaned"],
                        x="Default.Payment.Next.Month",
                        y="Limit_Bal",
                        labels={"Default.Payment.Next.Month": "Default (0 = No, 1 = Yes)",
                                "Limit_Bal": "Credit Limit"})
fig_target_box_title = "Credit Limit vs. Default Payment Next Month"


#%%
############
# Dash App #
############

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.FLATLY])

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
                        dbc.NavLink("Raw Data", 
                                    href="/raw-data", 
                                    active="exact", 
                                    style={"color": "black"}
                                    ),
                        dbc.NavLink("Clean Data", 
                                    href="/clean-data", 
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
    
#######################################
# Define the layout for the Data page #
#######################################
def clean_data_page_layout():
    return dbc.Container(
        fluid=True,
        className="py-0", # Remove vertical padding
        children=[
            
            # First row
            dbc.Row([
                
                # Left column: Control Panel
                dbc.Col(
                    
                    dbc.Card(
                        dbc.CardBody([
                            
                            dbc.Label("Distribution of Continuous Feature",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="hist-feature-dropdown",
                                options=sorted(continuous_var),
                                value=sorted(continuous_var)[0],
                                clearable=False,
                                className="mb-3"
                                ),
                                
                                dbc.Label("Target vs. Categorical Feature",
                                          className="label-style"),
                                dcc.Dropdown(
                                    id="cat-dropdown",
                                    options=[
                                        {"label": "Sex", "value": "Sex"},
                                        {"label": "Education", "value": "Education"},
                                        {"label": "Marriage", "value": "Marriage"}
                                        ],
                                    value="Sex",
                                    clearable=False,
                                    className="mb-3"
                                    ),
                                
                                dbc.Label("Continuous Feature by Default Status",
                                          className="label-style"),
                                dcc.Dropdown(
                                    id="feature-hist-dropdown",
                                    options=sorted(continuous_var),
                                    value=sorted(continuous_var)[0],
                                    clearable=False,
                                    className="mb-3"
                                    ),
                                
                                dbc.Label("Scatterplot of Continuous Features",
                                          className="label-style"),
                                dcc.Dropdown(
                                    id="feature-scat-dropdown",
                                    options=sorted(continuous_var),
                                    value=sorted(continuous_var)[0],
                                    clearable=False)
                                ]),
                                className="bg-primary"
                            ),
                    width=2 # 2 out of 12
                    ),    
                
                # Middle column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Default Payment Next Month", #id="target-dist-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }
                                       ),
                        dbc.CardBody([
                            dcc.Graph(figure=target_bar_plot(default_data, "Cleaned"))
                            ])
                            ], 
                        className="h-100")
                        ], 
                    width=5
                    ),
                    
                # Right column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="hist-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="hist-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=5)
                ]),
            
            # Second row
            dbc.Row([
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="cat-dist-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="cat-dist")
                            ])
                        ],
                        className="h-100")
                    ],
                    width={"size": 5, "offset": 2})
                
                ]),
            
            # Third row
            dbc.Row([
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="feature-hist-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="feature-hist")
                            ])
                        ],
                        className="h-100")
                    ],
                    width={"size": 5, "offset": 2}
                    ),
                
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="feature-scatter-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="feature-scatterplot")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=5
                    )
                
                ])
            
            ])

#########################################
# Define the layout for the Models page #
#########################################
def models_page_layout():
    return dbc.Container(
        fluid=True,
        className="py-0",
        children=[
            
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            
                            dbc.Label("Choose model",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=list(models.keys()),
                                value=list(models.keys())[0],
                                clearable=False,
                                className="mb-3"
                                )
                            ]),
                        className="bg-primary"
                        ),
                    width=2
                    ),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Classification Report",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="classification-report",
                                columns=[],
                                data=[],
                                style_table={"border": "1px solid #ccc"},
                                style_cell={
                                    "textAlign": "center",
                                    "border": "1px solid #ccc",
                                    "padding": "5px"
                                    },
                                style_header={
                                    "backgroundColor": "gray",
                                    "color": "white",
                                    "border": "1px solid #ccc",
                                    "fontWeight": "bold"
                                    }
                                ),
                            html.Div(id="summary-class-report", className="mt-3")
                            ]),
                        ],
                        className="h-100"),
                    
                    ],
                    width=5
                    ),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confusion Matrix",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="conf-matrix-plot")
                            ])
                        ],
                        className="h-100")
                    ], width=5
                    )
                ]),
            
            dbc.Row([
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="roc-curve-header",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="roc-curve-model")
                            ])
                        ],
                        className="h-100")
                    ], width={"size": 5, "offset": 2}
                    ),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Feature Importance",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="feature-model",
                                      config={"responsive": True})
                            ])
                        ],
                        className="h-100")
                ], width=5)
            ])
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
    - If the pathname is '/raw-data' the raw data content is shown.
    - If the pathname is '/clean-data' the cleaned data content is shown.
    - If the pathname is '/models' results of the models are shown.
    - If the pathname is '/model-comparison', comparison of models is shown.
    - Otherwise, a 404 error message is displayed.
    """
    if pathname == "/":
        return dcc.Location(pathname="/raw-data", id="redirect")
    elif pathname == "/raw-data":
        return html.Div([
            html.H3("This is the raw data page", style={"textAlign": "center"})
            ])
    elif pathname == "/clean-data":
        return clean_data_page_layout()
    elif pathname == "/models":
        return models_page_layout()
    elif pathname == "/model-comparison":
        return html.Div([
            html.H3("This is the model comparisons page", style={"textAlign": "center"})
            ])
    else:
        return html.Div([
            html.H3("404: Page not found", style={"textAlign": "center"})])

###########################################################################
# Data Page: Callback to update graph selection based on radioitem choice #
###########################################################################
@app.callback(
    Output("hist-feature-title", "children"),
    Output("hist-feature", "figure"),
    Input("hist-feature-dropdown", "value")
    )
def update_target_graph(selected_option):
    
    fig_feature_title = f"{selected_option} Distribution"
    fig_hist_feature = px.histogram(default_data["Cleaned"],
                                    x=selected_option)
    return fig_feature_title, fig_hist_feature
    
############################################################################################
# Data Page: Callback to update graph selection based on dropdown for categorical features #
############################################################################################
@app.callback(
    Output("cat-dist-title", "children"),
    Output("cat-dist", "figure"),
    Input("cat-dropdown", "value")
    )
def update_target_cat_graph(selected_option):
    
    count_cat = default_data["Cleaned"].groupby([selected_option, "Default.Payment.Next.Month"]).size().reset_index(name="count")
    count_cat = count_cat.rename(columns={"Default.Payment.Next.Month": "Default Payment"})
    count_cat["Default Payment"] = count_cat["Default Payment"].astype(str)
    cat_title = f"Default Count by {selected_option}"
    
    if selected_option == "Sex":
        cat_labels = {"Sex": "Sex (1 = Male, 2 = Female)", "count": "Count"}
    elif selected_option == "Education":
        cat_labels = {"Education": "Education (1 = Grad. School, 2 = Uni, 3 = HS, 4 = Others, 5&6 = Unk.)", "count": "Count"}
    else:
        cat_labels = {"Marriage": "Marriage (0 = Unk, 1 = Married, 2 = Single, 3 = Others)", "count": "Count"}
        
    fig_cat_bar = px.bar(count_cat,
                         x=selected_option,
                         y="count",
                         color="Default Payment",
                         barmode="group",
                         labels=cat_labels)
    
    return cat_title, fig_cat_bar

###########################################################################################
# Data Page: Callback to update graph selection based on dropdown for continuous features #
###########################################################################################
@app.callback(
    Output("feature-hist-title", "children"),
    Output("feature-hist", "figure"),
    Input("feature-hist-dropdown", "value")
    )
def update_feature_hist(selected_option):
    
    fig_feature_title = f"{selected_option} Distribution by Default Status"
    fig_feature_hist = px.histogram(default_data["Cleaned"],
                                    x=selected_option,
                                    color="Default.Payment.Next.Month",
                                    barmode="overlay",
                                    histnorm="density",
                                    opacity=0.25,
                                    labels={"Default.Payment.Next.Month": "Default Payment", f"{selected_option}": f"{selected_option}"})
    
    return fig_feature_title, fig_feature_hist

##########################################################
# Data Page: Callback to update scatterplots of features #
##########################################################
@app.callback(
    Output("feature-scatter-title", "children"),
    Output("feature-scatterplot", "figure"),
    Input("feature-scat-dropdown", "value")
    )
def update_feature_scatterplot(selected_option):
    
    df_scatter = default_data["Cleaned"]
    df_scatter["Default Payment"] = default_data["Cleaned"]["Default.Payment.Next.Month"].astype(str)
    fig_scatter_title = f"Limit_Bal vs. {selected_option}"

    fig_scatter = px.scatter(
        df_scatter,
        x="Limit_Bal",
        y=selected_option,
        color="Default Payment",
        labels={
            "Limit_Bal": "Credit Limit",
            selected_option: selected_option,
            "Default Payment": "Default Payment"
        }
    )

    # Reverse the order of the internal traces so that "1" is drawn last and placed on top of "0"
    fig_scatter.data = fig_scatter.data[::-1]

    return fig_scatter_title, fig_scatter

###########################################################################
# Models Page: Callbacks to update classification report and summary text #
###########################################################################
@app.callback(
    Output("classification-report", "data"),
    Output("classification-report", "columns"),
    Output("summary-class-report", "children"),
    Input("model-dropdown", "value")
    )
def update_class_report(selected_value):
    
    ##########################
    # 1) Build the dataframe #
    ##########################
    df_class_report = pd.DataFrame(eval_dict[selected_value]["classification_report"])
    
    # Prepare for the DataTable
    df_for_table = df_class_report.reset_index().rename(columns={"index": "Metric", "accuracy": "Accuracy", "macro avg": "Macro Avg", "weighted avg": "Weighted Avg"})
    df_for_table = round(df_for_table, 2)
    
    # Convert to DataTable format
    table_data = df_for_table.to_dict("records")
    table_columns = [{"name": col, "id": col} for col in df_for_table.columns]
    
    ###################
    # 2) Summary text #
    ###################
    # transpose for row/column access
    df_t = df_class_report.T 
    accuracy = df_t.loc["accuracy", "precision"]
    f1_class_0 = df_t.loc["0", "f1-score"]
    f1_class_1 = df_t.loc["1", "f1-score"]
    
    if f1_class_0 > f1_class_1:
        overall = "Overall, the model seems to handle Class 0 better than Class 1"
    else:
        overall = "Overall, the model seems to handle Class 1 better than Class 0"
    
    summary_text = dbc.Alert([
        html.H4(f"Model: {selected_value}", className="alert-heading"),
        html.P(f"Accuracy: {accuracy:.2f}"),
        html.P(f"F1-Score (Class 0): {f1_class_0:.2f}"),
        html.P(f"F1-Score (Class 1): {f1_class_1:.2f}"),
        html.Hr(),
        html.P(f"{overall}")
        ],
        color="info"
        )
    
    #####################
    # 3) Return outputs #
    #####################
    return table_data, table_columns, summary_text
    
####################################################
# Models Page: Callback to update confusion matrix #
####################################################
@app.callback(
    Output("conf-matrix-plot", "figure"),
    Input("model-dropdown", "value")
    )
def update_conf_matrix(selected_value):
    
    fig_cm = px.imshow(eval_dict[selected_value]["confusion_matrix"],
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual"),
                       x=["Not Default (pred)", "Default (pred)"], # Class labels on x-axis
                       y=["Not Default (actual)", "Default (actual)"], # ... y-axis
                       color_continuous_scale="Blues")
    
    fig_cm.update_layout(coloraxis_showscale=False)
    
    # Ensures axis ticks appear only at 0 and 1, rather than fractional values
    fig_cm.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["0", "1"])
    fig_cm.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["0", "1"])
    
    return fig_cm

#############################################
# Models Page: Callback to update ROC Curve #
#############################################
@app.callback(
    Output("roc-curve-header", "children"),
    Output("roc-curve-model", "figure"),
    Input("model-dropdown", "value")
    )
def update_roc_curve_model(selected_value):
    
    # False positive rate, true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, eval_dict[selected_value]["y_proba"])
    score = eval_dict[selected_value]["auc_score"]
    title_header = f"ROC Curve (AUC={score:.4f})"
    
    fig_roc = px.area(x=fpr,
                      y=tpr,
                      labels=dict(
                          x="False Positive Rate",
                          y="True Positive Rate"
                          ))
    
    fig_roc.add_shape(type="line",
                      line=dict(dash="dash"),
                      x0=0, x1=1, y0=0, y1=1)
    
    return title_header, fig_roc

######################################################
# Models Page: Callback to update feature importance #
######################################################
@app.callback(
    Output("feature-model", "figure"),
    Input("model-dropdown", "value")
    )
def update_feature_importance(selected_value):
    
    if selected_value == "Logistic Regression":
        sorted_val = "Abs Coefficient"
        y_vars = ["Abs Coefficient", "Odds Ratio"]
    elif selected_value == "Neural Network":
        sorted_val = "Importance Mean"
        y_vars = ["Importance Mean", "Importance Std"]
    else:
        sorted_val = "Importance"
        y_vars = "Importance"
        
    df_long = pd.melt(eval_dict[selected_value]["feature_importance"].sort_values(sorted_val, ascending=False).head(10),
                      id_vars="Feature",
                      value_vars=y_vars,
                      var_name="Metric",
                      value_name="Value")
    
    fig_fi = px.bar(df_long,
                    x="Value",
                    y="Feature",
                    color="Metric",
                    orientation="h",
                    barmode="group")
    
    return fig_fi

# Run the app    
if __name__ == '__main__': #http://127.0.0.1:8050/
    app.run_server(debug=True)