# RandomForest model training
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from train_model import train_model

def rf_train(df_model, X_trainval, y_trainval, cv_method: str):
    # RF base model
    rf_base = make_pipeline(
        SimpleImputer(strategy='mean'),
        RandomForestRegressor(random_state=42)
    )
    
    # possible model hypermeters
    param_grid_rf = {
        'randomforestregressor__n_estimators': [50, 100, 150], #[10, 25, 50, 100, 200, 500],
        'randomforestregressor__max_depth': [None, 10, 20],
        'randomforestregressor__min_samples_split': [2, 5, 10], # [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'randomforestregressor__min_samples_leaf': [1, 2, 4], # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'randomforestregressor__bootstrap': [True],
    }

    # cross validation method
    if cv_method == 'logo':
        # Leave one group out cross-validator
        groups = df_model['sub_id']
        logo = LeaveOneGroupOut()
        cv = logo.split(X_trainval, y_trainval, groups=groups)
    elif cv_method == 'gkf':
        # K-fold cross validator
        trainval_mask = ~df_model['is_temporal_test']
        group_trainval = df_model.loc[trainval_mask, ['sub_id', 'run_id']]
        gkf = GroupKFold(n_splits=5)
        cv = gkf.split(X_trainval, y_trainval, groups=group_trainval)

    best_model, best_params = train_model(X_trainval, y_trainval, rf_base, param_grid_rf, cv)
    
    return best_model, best_params
