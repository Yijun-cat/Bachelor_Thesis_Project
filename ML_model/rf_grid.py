# RandomForest parameter setting before grid search
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestRegressor

def rf_grid(df_model, X_trainval, y_trainval, cv_method: str):
    # RF base model
    base_model_rf = RandomForestRegressor(random_state=42)

    # possible model hypermeters
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        #'max_depth': [None, 10, 20],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [2, 3, 4],
        'bootstrap': [True],
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
    
    return base_model_rf, param_grid_rf, cv
