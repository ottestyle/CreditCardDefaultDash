import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Output, Input, no_update
from sklearn.metrics import roc_curve

def register_model_comparison_callbacks(app, eval_dict, y_test):
    # Combined ROC curve
    @app.callback(
        Output("combined-roc-curve", "figure"),
        Input("url", "pathname")
    )
    def update_combined_roc_curves(pathname):
        # Only updates if user is on the model comparison page
        if pathname != "/model-comparison":
            return no_update

        fig = go.Figure()
        # Looping through each model and plotting its ROC curve
        for model_name, metrics in eval_dict.items():
            y_proba = metrics["y_proba"]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (AUC: {round(metrics['auc_score'], 3)})"
                )
            )
        fig.update_layout(
            title="ROC Curves for All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white"
        )
        return fig

    # Combined Confusion Matrices
    @app.callback(
        Output("combined-confusion-matrix", "figure"),
        Input("url", "pathname")
    )
    def update_combined_confusion_matrices(pathname):
        if pathname != "/model-comparison":
            return no_update
        
        models = list(eval_dict.keys())
        n_models = len(models)

        # Subplots with one column per model
        fig = make_subplots(rows=1, cols=n_models, subplot_titles=models)
        
        # Looping through each model and create a heatmap
        for col_index, model_name in enumerate(models):
            metrics = eval_dict[model_name]
            cm = metrics["confusion_matrix"]
            trace = go.Heatmap(
                z=cm,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale="Blues",
                showscale=False,
                text=cm,
                texttemplate="%{text}"
            )
            fig.add_trace(trace, row=1, col=col_index+1)

        fig.update_layout(template="plotly_white")
        return fig

    # Combined Feature Importance
    @app.callback(
        Output("combined-feature-importance", "figure"),
        Input("url", "pathname")
    )
    def update_combinded_feature_importance(pathname):
        if pathname != "/model-comparison":
            return no_update
        
        models = list(eval_dict.keys())
        n_models = len(models)
        
        subplot_titles = []
        for model_name in models:
            df_fi = eval_dict[model_name]["feature_importance"]
            
            if "Abs Coefficient" in df_fi.columns:
                subplot_titles.append(f"Feature Abs Coefficient for {model_name}")
            elif "Importance" in df_fi.columns:
                subplot_titles.append(f"Feature Importances for {model_name}")
            elif "Importance Mean" in df_fi.columns:
                subplot_titles.append(f"Permutation Importances for {model_name}")
            else:
                subplot_titles.append(f"Feature Importances for {model_name}")
        
        fig = make_subplots(rows=1, cols=n_models, subplot_titles=subplot_titles)
        
        # Looping through each model and creating a horizontal bar chart
        for i, model_name in enumerate(models):
            df_fi = eval_dict[model_name]["feature_importance"]
            
            if "Abs Coefficient" in df_fi.columns:
                importance_col = "Abs Coefficient"
            elif "Importance" in df_fi.columns:
                importance_col = "Importance"
            elif "Importance Mean" in df_fi.columns:
                importance_col = "Importance Mean"
            else:
                importance_col = df_fi.columns[1]
                
            df_fi = df_fi.sort_values(by=importance_col, ascending=False).head(15)
        
            trace = go.Bar(
                x=df_fi[importance_col],
                y=df_fi["Feature"],
                orientation="h",
                name=model_name
                )
            
            # Adding the trace to the subplot
            fig.add_trace(trace, row=1, col=i+1)
            
            # Reversing the y-axis
            fig.update_yaxes(autorange="reversed", row=1, col=i+1)
        
        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=600
            )
        return fig

