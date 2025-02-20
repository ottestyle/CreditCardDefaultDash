import os
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from dash import Dash, html, Input, Output, dcc
import dash_bootstrap_components as dbc
import plotly.io as pio
import plotly.express as px

pio.renderers.default="browser"

# Load data
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# Change directory to app, modules and css 
os.chdir(os.environ["DEFAULT_OF_CREDIT_CARD_CLIENTS"])

from data_processing import preprocess_data
from evaluation import evaluate_model

eval_dict = {}

# Preprocess data
df_default, X, Y = preprocess_data(path)
df_default.columns = [x.lower().title() for x in df_default.columns]

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
continuous_var = ["Age", "Bill_Amt1", "Bill_Amt2", "Bill_Amt3", "Bill_Amt4", "Bill_Amt5", "Bill_Amt6", "Pay_Amt1",
                  "Pay_Amt2", "Pay_Amt3", "Pay_Amt4", "Pay_Amt5", "Pay_Amt6", "Total_Bill_Amt", "Credit_Utilization",
                  "Avg_Monthly_Utilization", "Bill_Trend", "Age_Limit_Interaction"]

# Static plots for distribution of target as well target vs. credit limit
count_target_data = df_default["Default.Payment.Next.Month"].value_counts().sort_index().reset_index()

fig_target_bar = px.bar(count_target_data, 
                        x="Default.Payment.Next.Month",
                        y="count",
                        title="Default Payment Next Month",
                        labels={
                            "Default.Payment.Next.Month": "Default (0 = No, 1 = Yes)",
                            "count": "Count"},
                        width=800,
                        height=500)

fig_target_box = px.box(df_default,
                        x="Default.Payment.Next.Month",
                        y="Limit_Bal",
                        title="Credit Limit vs. Default Payment Next Month",
                        labels={
                            "Default.Payment.Next.Month": "Default (0 = No, 1 = Yes)",
                            "Limit_Bal": "Credit Limit"
                            },
                        width=800,
                        height=500)

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
        # Keeps track of the URL so we can display different content based on it
        dcc.Location(id="url", refresh=False),
        
        # Header row with H1 in the top left, gray background and white text
        dbc.Row(
            dbc.Col(
                html.H1("Default of Credit Card Clients in Taiwan",
                        className="text-white",
                        style={
                            "backgroundColor": "gray",
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
    
#######################################
# Define the layout for the Data page #
#######################################
def data_page_layout():
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
                            
                            dbc.Label("Distribution"),
                            dcc.RadioItems(
                                id="target-radio",
                                options=[
                                    {"label": "Target Bar Chart", "value": "target-bar"},
                                    {"label": "Target vs. Credit Limit", "value": "target-box"}],
                                value="target-bar",
                                className="mb-3" # Spacing
                                ),
                                
                                dbc.Label("Target vs. Categorical Feature"),
                                dcc.Dropdown(
                                    id="cat-dropdown",
                                    options=[
                                        {"label": "Sex", "value": "Sex"},
                                        {"label": "Education", "value": "Education"},
                                        {"label": "Marriage", "value": "Marriage"}
                                        ],
                                    value="Sex",
                                    className="mb-3"
                                    ),
                                
                                dbc.Label("Histogram of a Continuous Feature"),
                                dcc.Dropdown(
                                    id="feature-hist-dropdown",
                                    options=sorted(continuous_var),
                                    value=sorted(continuous_var)[0],
                                    className="mb-3"
                                    ),
                                
                                dbc.Label("Scatterplot of a Continuous Feature"),
                                dcc.Dropdown(
                                    id="feature-scat-dropdown",
                                    options=sorted(continuous_var),
                                    value=sorted(continuous_var)[0])
                                ]),
                                className="bg-light" # Greyish background
                            ),
                    width=2 # 2 out of 12
                    ),    
                
                # Middle column
                dbc.Col([
                    dcc.Graph(id="target-dist")
                    ],
                    width=5
                    ),
                    
                # Right column
                dbc.Col([
                    dcc.Graph(id="cat-dist")
                    ],
                    width=5)

                ]),
            
            # Second row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="feature-hist")
                    ],
                    width={"size": 5, "offset": 2}
                    ),
                dbc.Col([
                    dcc.Graph(id="feature-scatterplot")
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
                            
                            dbc.Label("Choose model"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=list(models.keys()),
                                value=list(models.keys())[0],
                                className="mb-3"
                                )
                            ]),
                        className="bg-light"
                        ),
                    width=2
                    )
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
    - If the pathname is "/data" the cleaned data content is shown.
    - If the pathname is '/models' results of the models are shown.
    - If the pathname is '/model-comparison', comparison of models is shown.
    - Otherwise, a 404 error message is displayed.
    """
    if pathname == "/":
        return dcc.Location(pathname="/data", id="redirect")
    elif pathname == "/data":
        return data_page_layout()
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
    Output("target-dist", "figure"),
    Input("target-radio", "value")
    )
def update_target_graph(selected_option):
    
    if selected_option == "target-bar":
        return fig_target_bar
    elif selected_option == "target-box":
        return fig_target_box
    
############################################################################################
# Data Page: Callback to update graph selection based on dropdown for categorical features #
############################################################################################
@app.callback(
    Output("cat-dist", "figure"),
    Input("cat-dropdown", "value")
    )
def update_target_cat_graph(selected_option):
    
    count_cat = df_default.groupby([selected_option, "Default.Payment.Next.Month"]).size().reset_index(name="count")
    count_cat = count_cat.rename(columns={"Default.Payment.Next.Month": "Default Payment"})
    count_cat["Default Payment"] = count_cat["Default Payment"].astype(str)
    
    if selected_option == "Sex":
        cat_labels = {"Sex": "Sex (1 = Male, 2 = Female)", "count": "Count"}
    elif selected_option == "Education":
        cat_labels = {"Education": "Education (1 = Grad. School, 2 = Uni, 3 = HS, 4 = Others, 5&6 = Unk.)", "count": "Count"}
    else:
        cat_labels = {"Marriage": "Marriage (0 = Unk, 1 = Married, 2 = Single, 3 = Others)", "count": "Count"}
    
    return px.bar(count_cat,
                  x=selected_option,
                  y="count",
                  color="Default Payment",
                  barmode="group",
                  title=f"Default Count by {selected_option}",
                  labels = cat_labels,
                  width=800,
                  height=500)

###########################################################################################
# Data Page: Callback to update graph selection based on dropdown for continuous features #
###########################################################################################
@app.callback(
    Output("feature-hist", "figure"),
    Input("feature-hist-dropdown", "value")
    )
def update_feature_hist(selected_option):
    
    return px.histogram(df_default,
                        x=selected_option,
                        color="Default.Payment.Next.Month",
                        barmode="overlay",
                        histnorm="density",
                        opacity=0.25,
                        title=f"{selected_option} Distribution by Default Status",
                        labels={"Default.Payment.Next.Month": "Default Payment", f"{selected_option}": f"{selected_option}"},
                        width=800,
                        height=500
                        )

##########################################################
# Data Page: Callback to update scatterplots of features #
##########################################################
@app.callback(
    Output("feature-scatterplot", "figure"),
    Input("feature-scat-dropdown", "value")
    )
def update_feature_scatterplot(selected_option):
    
    df_scatter = df_default
    df_scatter["Default Payment"] = df_default["Default.Payment.Next.Month"].astype(str)

    fig_scatter = px.scatter(
        df_scatter,
        x="Limit_Bal",
        y=selected_option,
        color="Default Payment",  
        title=f"Limit_Bal vs. {selected_option}",
        labels={
            "Limit_Bal": "Credit Limit",
            selected_option: selected_option,
            "Default Payment": "Default Payment"
        },
        width=800,
        height=500
    )

    # Reverse the order of the internal traces so that "1" is drawn last and placed on top of "0"
    fig_scatter.data = fig_scatter.data[::-1]

    return fig_scatter

# Run the app    
if __name__ == '__main__': #http://127.0.0.1:8050/
    app.run_server(debug=True)