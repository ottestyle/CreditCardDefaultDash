import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

def model_comparison_layout(eval_dict):
    """layout for the model comparison page""" 
    # Summary table
    table_data = []
    for model_name, metrics in eval_dict.items():
        table_data.append({
            "Model": model_name,
            "ROC AUC": round(metrics["auc_score"], 3),
            "Accuracy": round(metrics["classification_report"]["accuracy"], 3),
            "F1-Score (Class 0)": round(metrics["classification_report"]["0"]["f1-score"], 3),
            "F1-Score (Class 1)": round(metrics["classification_report"]["1"]["f1-score"], 3)
        })
    
    return dbc.Container(
        fluid=True,
        children=[
            
            dbc.Row([
                dbc.Col(html.H2("Model Comparison"), width=12),
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                        }),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="model-summary-table",
                                data=table_data,
                                columns=[{"name": k, "id": k} for k in table_data[0].keys()],
                                style_table={"overflowX": "auto", "border": "1px solid #ccc"},
                                style_cell={"textAlign": "center", "border": "1px solid #ccc", "padding": "5px"},
                                style_header={
                                    "backgroundColor": "gray",
                                    "color": "white",
                                    "border": "1px solid #ccc",
                                    "fontWeight": "bold"
                                }
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ROC Curves",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dcc.Graph(id="combined-roc-curve")
                        ])
                    ], width=6)
                ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Combined Confusion Matrices",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dcc.Graph(id="combined-confusion-matrix")
                        ])
                    ], width=12)
                ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Combined Feature Importance",
                                       className="bg-primary text-white",
                                       style={
                                           "fontSize": "20px",
                                           "fontWeight": "bold"
                                           }),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(
                                    dcc.Graph(id="combined-feature-importance"),
                                    width=12
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            
            ]
        )
    