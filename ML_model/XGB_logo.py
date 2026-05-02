from build_dataframe import construct_df
import cupy as cp
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from XGB_train import xgb_train

subjects = ['03', '04', '05', '06', '08', 11, 12,
            13, 15, 16, 17, 18, 19, 20, 22, 23, 24,
            27, 31, 32, 33, 35, 37, 38, 43
]
    
# get dataframe, features and targets columns
df_model, feat_cols, target_cols = construct_df(subjects)
X = df_model[feat_cols].to_numpy()
y = df_model[target_cols].to_numpy()

best_model, best_params = xgb_train(df_model, X, y, cv_method='logo')

# evaluate model performance using the best model 
logo = LeaveOneGroupOut()
groups = df_model['sub_id']
# lists for per fold performance metrics
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

print("===XGboost Model Performance (subject-level generalization)===")
print("Mean MAE:", np.mean(mae))
print("Mean RMSE:", np.mean(rmse))
print("Mean R2:", np.mean(r2))