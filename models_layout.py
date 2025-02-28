import dash_bootstrap_components as dbc
from dash import dcc, dash_table, html

def models_layout(models):
    """Layout for the models page"""
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