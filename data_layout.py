# app/layouts/data_layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def data_layout(default_data):
    """Layout for the data page"""
    return dbc.Container(
        fluid=True,
        className="py-0", 
        children=[
            
            # Header and version toggle
            dbc.Row([
                dbc.Col(html.H2("Data Overview & EDA"), width=12),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Label("Select Data Version", className="label-style"),
                            dcc.RadioItems(id="data-version-toggle",
                                           options=[
                                               {"label": "Raw Data", "value": "Raw"},
                                               {"label": "Cleaned Data", "value": "Cleaned"}
                                               ],
                                           value="Raw",
                                           inline=True,
                                           className="mb-3",
                                           labelStyle={"margin-right": "10px"},
                                           style={"color": "white"}
                                           )
                            ]),
                        className="bg-primary"
                        ),
                    width=2
                    )
                ], className="mb-4"),
            
            # Data preview and descriptive statistics
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Descriptive Statistics",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody(dash_table.DataTable(
                            id="summary-table",
                            data=[],
                            columns=[],
                            style_table={
                                "overflowX": "auto", 
                                "border": "1px solid #ccc"
                                },
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
                            ))
                    ]),
                    width=12
                )
            ], className="mb-4"),
            
            # Distribution plots -> one dropdown for both histogram and boxplot
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Distribution Analysis",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Dropdown(id="dist-feature-dropdown",
                                         options=[],
                                         clearable=False,
                                         style={"width": "300px"}
                                         ),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="histogram-graph"), width=6),
                                dbc.Col(dcc.Graph(id="boxplot-graph"), width=6)
                            ])
                        ])
                    ]),
                    width=12
                )
            ], className="mb-4"),
            
            # Multicollinearity -> VIF table and correlation heatmap
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("VIF Analysis",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody(dash_table.DataTable(
                            id="vif-table",
                            style_table={"overflowX": "auto"},
                            style_data_conditional=[{"if": {"filter_query": "{VIF} > 10"}, "backgroundColor": "#FFCCCC"}]
                        ))
                    ]),
                    width=6
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Correlation Heatmap",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody(dcc.Graph(id="corr-heatmap"))
                    ]),
                    width=6
                )
            ], className="mb-4"),
            
            # Categorical analysis -> bar chart by default status
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Categorical Analysis - Bar Chart",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dcc.Dropdown(id="cat-feature-dropdown",
                                         options=[],
                                         value=[],
                                         clearable=False,
                                         style={"width": "300px"}
                                    ),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id="cat-pie-chart"), width=6),
                                dbc.Col(dcc.Graph(id="cat-bar-chart"), width=6)
                                ])
                        ])
                    ]),
                    width=12
                )
            ])
        ])