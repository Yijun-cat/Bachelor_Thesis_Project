# Catboost parameter setting
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor

def cat_grid(df_model, X_trainval, y_trainval, cv_method: str):
    base_estimator = CatBoostRegressor(
        loss_function='RMSE',
        random_seed = 42,
        # task_type = "GPU",
        task_type = "CPU",
        # devices = "0",
        verbose = False,
        thread_count = 1,
    )

    base_model_cat = MultiOutputRegressor(base_estimator)

    param_grid_cat = {
        "estimator__depth": [2, 3],
        "estimator__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "estimator__l2_leaf_reg": [6, 10]
    }

    if cv_method == "logo":
        groups = df_model["sub_id"]
        logo = LeaveOneGroupOut()
        cv = logo.split(X_trainval, y_trainval, groups=groups)

    elif cv_method == "gkf":
        trainval_mask = ~df_model["is_temporal_test"]
        group_trainval = (
            df_model.loc[trainval_mask, "sub_id"].astype(str)
            + "_"
            + df_model.loc[trainval_mask, "run_id"].astype(str)
        )
        gkf = GroupKFold(n_splits=5)
        cv = gkf.split(X_trainval, y_trainval, groups=group_trainval)
    else:
        raise ValueError("cv_method must be 'logo' or 'gkf'")
    
    return base_model_cat, param_grid_cat, cv


