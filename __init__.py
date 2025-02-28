from .clean_callbacks import register_clean_callbacks
from .models_callbacks import register_models_callbacks

def register_callbacks(app, default_data, eval_dict, y_test):
    register_clean_callbacks(app, default_data)
    register_models_callbacks(app, eval_dict, y_test)