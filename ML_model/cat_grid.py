# CatBoost hyperparameters to be explored and cross validation methods
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor

def cat_grid(df_model, X_trainval, y_trainval, cv_method: str):
    # define base CatBoost model
    base_estimator = CatBoostRegressor(
        loss_function='RMSE',
        random_seed = 42,
        task_type = "CPU",
        verbose = False,
        thread_count = 1,
    )

    # warp base model in MultiOutputRegressor from scikit-learn
    base_model_cat = MultiOutputRegressor(base_estimator)

    # explored hyperparameters
    param_grid_cat = {
        "estimator__depth": [2, 3],
        "estimator__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "estimator__l2_leaf_reg": [6, 10]
    }

    # cross validation methods
    if cv_method == "logo": # use leave-one-group-out in subject-level generalization
        groups = df_model["sub_id"]
        logo = LeaveOneGroupOut()
        cv = logo.split(X_trainval, y_trainval, groups=groups)
    elif cv_method == "gkf":# use K-fold cross validation in temporal generalization
        # train and validation set
        trainval_mask = ~df_model["is_temporal_test"]
        # same subject ID and run number is in the same group
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


