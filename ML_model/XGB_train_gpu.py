import os
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from train_model_gpu import train_model_gpu
from sklearn import set_config
set_config(array_api_dispatch=True)

def xgb_train_gpu(df_model, X_trainval, y_trainval, cv_method: str):
    # XGB base model
    xgb_base =XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=20,
        device = "cuda",
        multi_strategy = 'multi_output_tree'
    )

    # possible parameters
    param_grid_xgb = {
        'n_estimators': [25, 50, 100],
        "max_depth": [2, 3, 4, 5],
        "colsample_bytree": [0.25, 0.5],
        "booster": ["dart", "gbtree"]
    }
    
    # cross validation method
    if cv_method == 'logo': # Leave one group out cross-validator
        groups = df_model["sub_id"].to_pandas().to_numpy()  # NumPy 1D array
        cv = LeaveOneGroupOut()
    elif cv_method == 'gkf': # K-fold cross validator
        trainval_mask = ~df_model['is_temporal_test']
        group_trainval = (
            df_model.loc[trainval_mask, "sub_id"].astype(str)
            + "_"
            + df_model.loc[trainval_mask, "run_id"].astype(str)
        )
        group_trainval_np = group_trainval.to_pandas().to_numpy()
        cv = GroupKFold(n_splits=5)
        groups = group_trainval_np
    else:
        raise ValueError("cv_method must be 'logo' or 'gkf'")

    best_model, best_params = train_model_gpu(X_trainval, y_trainval, xgb_base, param_grid_xgb, cv, groups=groups, device="cuda")
    
    return best_model, best_params

