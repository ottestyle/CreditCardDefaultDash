import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, html
import dash_bootstrap_components as dbc
from sklearn.metrics import roc_curve

def register_models_callbacks(app, eval_dict, y_test):
    # Update classification report and summary text
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
        df_for_table = round(df_for_table, 3)
        
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
        
        return table_data, table_columns, summary_text
        
    # Update confusion matrix
    @app.callback(
        Output("conf-matrix-plot", "figure"),
        Input("model-dropdown", "value")
    )
    def update_conf_matrix(selected_value):
        
        cm = eval_dict[selected_value]["confusion_matrix"]
        
        heatmap = go.Heatmap(
            z=cm,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}"
            )
        fig = go.Figure(data=[heatmap])
        fig.update_layout(template="plotly_white")
        
        return fig

    # Update ROC curve
    @app.callback(
        Output("roc-curve-header", "children"),
        Output("roc-curve-model", "figure"),
        Input("model-dropdown", "value")
    )
    def update_roc_curve_model(selected_value):
        
        # False positive rate, true positive rate
        fpr, tpr, thresholds = roc_curve(y_test, eval_dict[selected_value]["y_proba"])
        score = eval_dict[selected_value]["auc_score"]
        title = f"ROC Curve (AUC={score:.3f})"
        
        fig = px.area(x=fpr,
                      y=tpr,
                      labels=dict(
                          x="False Positive Rate",
                          y="True Positive Rate"
                          ))
        
        fig.add_shape(type="line",
                      line=dict(dash="dash"),
                      x0=0, x1=1, y0=0, y1=1)
        
        return title, fig
    
    # Update feature importance plot
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
        
        fig = px.bar(df_long,
                     x="Value",
                     y="Feature",
                     color="Metric",
                     orientation="h",
                     barmode="group")
        
        fig.update_yaxes(autorange="reversed")
        
        return fig