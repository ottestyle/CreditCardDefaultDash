# app/callbacks/data_callbacks.py
import plotly.express as px
from dash import Input, Output

def generate_corr_heatmap(df, features):
    corr = df[features].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(title="Correlation Matrix")
    return fig

def register_data_callbacks(app, default_data, df_vif):
    # Descriptive statistics
    @app.callback(
        Output("summary-table", "data"),
        Output("summary-table", "columns"),
        Input("data-version-toggle", "value")
    )
    def update_summary_table(version):
        summary = default_data[version].describe().reset_index().rename(columns={"index": "Stats"})
        summary = round(summary, 0)
        summary_cols = [{"name": col, "id": col} for col in summary.columns]
        summary_data = summary.to_dict("records")
        return summary_data, summary_cols
    
    # Continuous feature dropdown
    @app.callback(
        Output("dist-feature-dropdown", "options"),
        Output("dist-feature-dropdown", "value"),
        Input("data-version-toggle", "value")
    )
    def update_continuous_feature_dropdown(version):
        features = {
            "Raw": ["Limit Bal", "Age", "Bill Amt1", "Bill Amt2", "Bill Amt3", "Bill Amt4", "Bill Amt5",
                    "Bill Amt6", "Pay Amt1", "Pay Amt2", "Pay Amt3", "Pay Amt4", "Pay Amt5", "Pay Amt6"],
            "Cleaned": ["Limit Bal", "Age", "Total Bill Amt", "Credit Utilization",
                        "Bill Trend", "Pay Trend", "Age Limit Interaction"]
        }
        
        # Convert to dropdown options
        options = [{"label": col, "value": col} for col in features[version]]
        
        # Set a default value
        default_value = options[0]["value"] if options else None

        return options, default_value
    
    # Continuous feature distribution
    @app.callback(
        Output("histogram-graph", "figure"),
        Output("boxplot-graph", "figure"),
        Input("dist-feature-dropdown", "value"),
        Input("data-version-toggle", "value")
    )
    def update_continuous_distribution_plots(feature, version):
        df = default_data[version]
        hist_fig = px.histogram(df, x=feature, nbins=30, title=f"{feature} ({version})")
        box_fig = px.box(df, y=feature, title=f"{feature} ({version})")
        return hist_fig, box_fig
    
    # VIF analysis
    @app.callback(
        Output("vif-table", "data"),
        Output("vif-table", "columns"),
        Input("data-version-toggle", "value")
    )
    def update_vif_table(version):
        # Use the cleaned data for VIF
        df = round(df_vif,2)
        cols = [{"name": col, "id": col} for col in df.columns]
        data = df.to_dict("records")
        return data, cols
    
    # Correlation heatmap
    @app.callback(
        Output("corr-heatmap", "figure"),
        Input("data-version-toggle", "value")
    )
    def update_corr_heatmap(version):
        df_corr = default_data[version]
        features = df_corr.columns
        fig = generate_corr_heatmap(df_corr, features)
        return fig
    
    # Categorical feature distribution
    @app.callback(
        Output("cat-pie-chart", "figure"),
        Output("cat-bar-chart", "figure"),
        Input("cat-feature-dropdown", "value"),
        Input("data-version-toggle", "value")
    )
    def update_categorical_charts(feature, version):
        df = default_data[version]
        
        # Pie chart
        count_pie = df[feature].value_counts().reset_index(name="count")
        title_pie = f"{feature} ({version})"
        
        # Bar chart
        count_bar = df.groupby([feature, "Default Payment Next Month"]).size().reset_index(name="count")
        count_bar["Default Payment Next Month"] = count_bar["Default Payment Next Month"].astype(str)
        title_bar = f"Default Count by {feature} ({version})"
        
        pie_labels = {
            "Sex": {"1": "Male", "2": "Female"},
            "Education": {"1": "Grad. School", "2": "Uni", "3": "HS", "4": "Others", "5&6": "Unknown"},
            "Marriage": {"0": "Unknown", "1": "Married", "2": "Single", "3": "Others"},
            "Max Delay": {"1": "Delay 1M", "2": "Delay 2M", "3": "Delay 3M", "4": "Delay 4M", "5": "Delay 5M", "6": "Delay 6M"}
            }
        
        bar_labels = {
            "Sex": {"Sex": "Sex (1 = Male, 2 = Female)", "count": "Count"},
            "Education": {"Education": "Education (1 = Grad. School, 2 = Uni, 3 = HS, 4 = Others, 5&6 = Unk.)", "count": "Count"},
            "Marriage": {"Marriage": "Marriage (0 = Unknown, 1 = Married, 2 = Single, 3 = Others)", "count": "Count"},
            "Max Delay": {"Max Delay": "1 = Delay 1M, ... , 6 = Delay 6M", "count": "Count"}
                    }
        # Create keys for Pay0 through Pay6
        for i in [0, 2, 3, 4, 5, 6]:
            pie_labels[f"Pay{i}"] = {
                "-2": "No consumption", "-1": "Paid in full", "0": "Use of revolving credit", 
                "1": "Delay 1M", "2": "Delay 2M", "3": "Delay 3M", 
                "4": "Delay 4M", "5": "Delay 5M", "6": "Delay 6M"}
            bar_labels[f"Pay{i}"] = {
                f"Pay{i}": "-2 = No consumption, -1 = Paid in full, 0 = Use of revolving credit, 1 = Delay 1M, ... , 8 = Delay 8M", 
                "count": "Count"
                }
        
        # Renaming the first column
        count_pie = count_pie.rename(columns={count_pie.columns[0]: "category"})
    
        # Mapping values to the labels
        count_pie["category"] = count_pie["category"].astype(str).map(pie_labels[feature])
    
        fig_pie = px.pie(count_pie, names="category", values="count", title=title_pie)
        fig_bar = px.bar(count_bar, x=feature, y="count", color="Default Payment Next Month", barmode="group", labels=bar_labels[feature], title=title_bar)
        return fig_pie, fig_bar
    
    # Categorical feature dropdown
    @app.callback(
        Output("cat-feature-dropdown", "options"),
        Output("cat-feature-dropdown", "value"),
        Input("data-version-toggle", "value")
    )
    def update_categorical_feature_dropdown(version):
        features = {
            "Raw": ["Sex", "Education", "Marriage", "Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6"],
            "Cleaned": ["Sex", "Education", "Marriage", "Pay0", "Pay2", "Pay3", "Pay4", "Pay5", "Pay6", "Max Delay"]
        }
        
        # Convert to dropdown options
        options = [{"label": col, "value": col} for col in features[version]]
        
        # Set a default value
        default_value = options[0]["value"] if options else None

        return options, default_value
