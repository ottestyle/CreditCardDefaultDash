# app/callbacks/__init__.py
from.data_callbacks import register_data_callbacks
from .models_callbacks import register_models_callbacks
from .model_comparison_callbacks import register_model_comparison_callbacks

def register_callbacks(app, default_data, df_vif, eval_dict, y_test):
    register_data_callbacks(app, default_data, df_vif)
    register_models_callbacks(app, eval_dict, y_test)
    register_model_comparison_callbacks(app, eval_dict, y_test)