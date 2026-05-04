from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from train_model import train_model

def xgb_grid(df_model, X_trainval, y_trainval, cv_method: str):
    # XGB base model
    base_model_xgb = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            device = "cpu",
            n_jobs = None, 
            multi_strategy = 'multi_output_tree'
        )
    
    # possible parameters
    param_grid_xgb = {
        "estimator__n_estimators": [10, 100],
        "estimator__max_depth": [2, 3, 4, 5],
        "estimator__colsample_bytree": [0.3, 0.4, 0.5],
        "estimator__booster": ["dart", "gbtree"]
    }

    # cross validation method
    if cv_method == 'logo': # Leave one group out cross-validator
        groups = df_model['sub_id']
        logo = LeaveOneGroupOut()
        cv = logo.split(X_trainval, y_trainval, groups=groups)
    elif cv_method == 'gkf': # K-fold cross validator
        trainval_mask = ~df_model['is_temporal_test']
        group_trainval = (
            df_model.loc[trainval_mask, "sub_id"].astype(str)
            + "_"
            + df_model.loc[trainval_mask, "run_id"].astype(str)
        )
        gkf = GroupKFold(n_splits=5)
        cv = gkf.split(X_trainval, y_trainval, groups=group_trainval)
    else:
        raise ValueError("cv_method must be 'logo' or 'gkf'")
    
    return base_model_xgb, param_grid_xgb, cv

