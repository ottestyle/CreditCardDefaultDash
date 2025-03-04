import plotly.express as px
from dash import Input, Output

def register_raw_callbacks(app, default_data):
    # Histogram for continuous features
    @app.callback(
        Output("raw-hist-feature-title", "children"),
        Output("raw-hist-feature", "figure"),
        Input("raw-hist-feature-dropdown", "value")
        )
    def update_hist_feature(selected_option):
        
        title = f"{selected_option} Distribution"
        fig = px.histogram(default_data["Raw"], x=selected_option)
        return title, fig
    
    # Boxplot for continouos features
    @app.callback(
        Output("raw-boxplot-feature-title", "children"),
        Output("raw-boxplot-feature", "figure"),
        Input("raw-boxplot-feature-dropdown", "value")
        )
    def update_boxplot_feature(selected_option):
        
        title = f"{selected_option} Distribution"
        fig = px.box(default_data["Raw"], y=selected_option)
        return title, fig
    
    # Bar chart for categorical features by default status
    @app.callback(
        Output("raw-barchart-feature-title", "children"),
        Output("raw-barchart-feature", "figure"),
        Input("raw-barchart-feature-dropdown", "value")
        )
    def update_target_cat_graph(selected_option):
        
        count_cat = default_data["Raw"].groupby([selected_option, "Default Payment Next Month"]).size().reset_index(name="count")
        count_cat = count_cat.rename(columns={"Default Payment Next Month": "Default Payment"})
        count_cat["Default Payment"] = count_cat["Default Payment"].astype(str)
        title = f"Default Count by {selected_option}"
        
        if selected_option == "Sex":
            cat_labels = {"Sex": "Sex (1 = Male, 2 = Female)", "count": "Count"}
        elif selected_option == "Education":
            cat_labels = {"Education": "Education (1 = Grad. School, 2 = Uni, 3 = HS, 4 = Others, 5&6 = Unk.)", "count": "Count"}
        else:
            cat_labels = {"Marriage": "Marriage (0 = Unk, 1 = Married, 2 = Single, 3 = Others)", "count": "Count"}
            
        fig = px.bar(count_cat,
                     x=selected_option, 
                     y="count",
                     color="Default Payment",
                     barmode="group",
                     labels=cat_labels)
        
        return title, fig
    
    # Histogram for continuous features by default status
    @app.callback(
        Output("raw-hist-density-feature-title", "children"),
        Output("raw-hist-density-feature", "figure"),
        Input("raw-hist-density-feature-dropdown", "value")
        )
    def update_feature_hist(selected_option):
        
        title = f"{selected_option} Distribution by Default Status"
        fig = px.histogram(default_data["Raw"],
                           x=selected_option,
                           color="Default Payment Next Month",
                           barmode="overlay",
                           histnorm="density",
                           opacity=0.25,
                           labels={"Default Payment Next Month": "Default Payment", f"{selected_option}": f"{selected_option}"})
        
        return title, fig
    
    # Scatterplot for continuous features vs credit limit
    @app.callback(
        Output("raw-scatter-feature-title", "children"),
        Output("raw-scatter-feature", "figure"),
        Input("raw-scatter-feature-dropdown", "value")
        )
    def update_feature_scatterplot(selected_option):
        
        df_scatter = default_data["Raw"]
        df_scatter["Default Payment"] = default_data["Cleaned"]["Default Payment Next Month"].astype(str)
        title = f"Limit Bal vs. {selected_option}"
    
        fig = px.scatter(df_scatter,
                         x="Limit Bal",
                         y=selected_option,
                         color="Default Payment",
                         labels={
                             "Limit Bal": "Credit Limit",
                             selected_option: selected_option,
                             "Default Payment": "Default Payment"
                             }
                         )
    
        # Reverse the order of the internal traces so that "1" is drawn last and placed on top of "0"
        fig.data = fig.data[::-1]
    
        return title, fig