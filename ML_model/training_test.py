# RandomForest subject-level generalization

from build_dataframe import construct_df
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from train_model import train_ml

subjects = ['03', '04', '05', '06', '08', 11, 12,
            13, 15, 16, 17, 18, 19, 20, 22, 23, 24,
            27, 31, 32, 33, 35, 37, 38, 43]
    
# get dataframe, features and targets columns
df_model, X, y = construct_df(['03', '04', '05'])

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
# Leave one group out cross-validator
groups = df_model['sub_id']
logo = LeaveOneGroupOut()
cv = logo.split(X, y, groups)

best_model, best_params = train_ml(X, y, rf_base, param_grid_rf, cv)

# evaluate model performance using the best model 
mae = []
rmse = []
r2 = []

for train_id, test_id in logo.split(X, y, groups):
    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]
    best_model.fit(X_train, y_train)
    y_pred_fold = best_model.predict(X_test)
    mae_fold = mean_absolute_error(y_test, y_pred_fold)
    rmse_fold = root_mean_squared_error(y_test, y_pred_fold)
    r2_fold = r2_score(y_test, y_pred_fold)
    mae.append(mae_fold)
    rmse.append(rmse_fold)
    r2.append(r2_fold)

print("===RF Model Performance (subject-level generalization)===")
print("Mean MAE:", np.mean(mae))
print("Mean RMSE:", np.mean(rmse))
print("Mean R2:", np.mean(r2))