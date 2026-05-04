import os
os.environ["SCIPY_ARRAY_API"] = "1"

from build_dataframe import construct_df
import numpy as np
import cupy as cp
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from ML_model.on_gpu.XGB_train_gpu import xgb_train_gpu
import cudf
from sklearn import set_config
set_config(array_api_dispatch=True)

subjects = ['03', '04', '05', '06', '08', 11, 12,
            13, 15, 16, 17, 18, 19, 20, 22, 23, 24,
            27, 31, 32, 33, 35, 37, 38, 43
]
    
# get dataframe, features and targets columns
df_model_pd, feat_cols, target_cols = construct_df(['03', '04', '05'], with_lag_feature=True)
df_model = cudf.from_pandas(df_model_pd)
# get features and targets as cuDF DataFrames
gdf_X = df_model[feat_cols]        
gdf_y = df_model[target_cols]      
# put feature and targets to GPU
# convert cuDF to CuPy (stays on GPU)
X = cp.array(gdf_X.to_cupy())
y = cp.array(gdf_y.to_cupy())

best_model, best_params = xgb_train_gpu(df_model, X, y, cv_method='logo')

# evaluate model performance using the best model 
logo = LeaveOneGroupOut()
groups = df_model_pd['sub_id'].to_numpy()
# lists for per fold performance metrics
mae = []
rmse = []
r2 = []

for train_id, test_id in logo.split(X, y, groups):
    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]
    best_model.fit(X_train, y_train)
    y_test = cp.asnumpy(y_test)
    #y_pred_fold = cp.asarray( best_model.predict(X_test) )
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