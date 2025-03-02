import dash_bootstrap_components as dbc
from dash import dcc
import plotly.express as px

def raw_data_layout(continuous_var, default_data):
    """Layout for the raw data page"""
    return dbc.Container(
        fluid=True,
        className="py-0", # Remove vertical padding
        children=[
            
            # First row
            dbc.Row([
                
                # First column: Control Panel
                dbc.Col(
                    
                    dbc.Card(
                        dbc.CardBody([
                            
                            dbc.Label("Histogram of a Continuous Feature",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="raw-hist-feature-dropdown",
                                options=sorted(continuous_var),
                                value=sorted(continuous_var)[0],
                                clearable=False,
                                className="mb-3"
                                ),
                            
                            dbc.Label("Boxplot of a Continuous Feature",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="raw-boxplot-feature-dropdown",
                                options=sorted(continuous_var),
                                value=sorted(continuous_var)[0],
                                clearable=False,
                                className="mb-3"
                                ),
                            
                            dbc.Label("Categorical Feature by Default Status",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="raw-barchart-feature-dropdown",
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
                                id="raw-hist-density-feature-dropdown",
                                options=sorted(continuous_var),
                                value=sorted(continuous_var)[0],
                                clearable=False,
                                className="mb-3"
                                ),
                                
                            dbc.Label("Scatterplot of Continuous Features",
                                      className="label-style"),
                            dcc.Dropdown(
                                id="raw-scatter-feature-dropdown",
                                options=sorted(continuous_var),
                                value=sorted(continuous_var)[0],
                                clearable=False)
                            ]),
                        className="bg-primary"
                            ),
                    width=2 # 2 out of 12
                    ),    
                
                # Second column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Default Payment Next Month",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }
                                       ),
                        dbc.CardBody([
                            dcc.Graph(figure=target_bar_plot(default_data, "Raw"))
                            ])
                            ], 
                        className="h-100")
                        ], 
                    width=3
                    ),
                    
                # Third column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="raw-hist-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="raw-hist-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=3),
                
                # Fourth column
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="raw-boxplot-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="raw-boxplot-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=3)
                ]),
            
            # Second row
            dbc.Row([
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="raw-barchart-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="raw-barchart-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width={"size": 3, "offset": 2}
                    ),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="raw-hist-density-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="raw-hist-density-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=3
                    ),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(id="raw-scatter-feature-title",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Graph(id="raw-scatter-feature")
                            ])
                        ],
                        className="h-100")
                    ],
                    width=3
                    )
                ]),
            ])

def target_bar_plot(df, data_state):
    count_target_data = df[data_state]["Default Payment Next Month"].value_counts().sort_index().reset_index()

    return px.bar(count_target_data,
                  x="Default Payment Next Month",
                  y="count",
                  labels={"Default Payment Next Month": "Default (0 = No, 1 = Yes)", "count": "Count"})