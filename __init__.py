# app/callbacks/__init__.py
from.data_callbacks import register_data_callbacks
from .models_callbacks import register_models_callbacks

def register_callbacks(app, default_data, df_vif, eval_dict, y_test):
    register_data_callbacks(app, default_data, df_vif)
    register_models_callbacks(app, eval_dict, y_test)