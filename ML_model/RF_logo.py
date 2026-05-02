# RandomForest subject-level generalization

from build_dataframe import construct_df
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from rf_grid import rf_grid
from train_model import train_model

subjects = [
    "03", "04", "05", "06", "08",
    "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "22",
    "23", "24", "27", "31", "32",
    "33", "35", "37", "38", "43"
]
    
# get dataframe, features and targets columns
df_model, feat_cols, target_cols = construct_df(subjects)
X = df_model[feat_cols].to_numpy()
y = df_model[target_cols].to_numpy()

# Hyper parameter tuning using grid search
base_model, param_grid, cv = rf_grid(df_model, X, y, cv_method='logo')
best_model, best_params = train_model(X, y, base_model, param_grid, cv)

# Evaluate best model performance 
mae_list =[]
rmse_list = []
r2_list = []
metrics = []
preds = []

logo = LeaveOneGroupOut()
groups = df_model['sub_id']

for train_id, test_id in logo.split(X, y, groups):
    X_train, X_test = X[train_id], X[test_id]
    y_train, y_test = y[train_id], y[test_id]
    subject_test = df_model.iloc[test_id]["sub_id"].iloc[0]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mae_fold = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    rmse_fold = root_mean_squared_error(y_test, y_pred, multioutput="raw_values")
    r2_fold = r2_score(y_test, y_pred, multioutput="raw_values")

    mae_list.append(mae_fold)
    rmse_list.append(rmse_fold)
    r2_list.append(r2_fold)
    metrics.append({
        "sub_id": subject_test,
        "MAE_glideslope": mae_fold[0],
        "MAE_localizer": mae_fold[1],
        "MAE_airspeed": mae_fold[2],
        "MAE_total": mae_fold[3],
        "RMSE_glideslope": rmse_fold[0],
        "RMSE_localizer": rmse_fold[1],
        "RMSE_airspeed": rmse_fold[2],
        "RMSE_total": rmse_fold[3],
        "R2_glideslope": r2_fold[0],
        "R2_localizer": r2_fold[1],
        "R2_airspeed": r2_fold[2],
        "R2_total": r2_fold[3],
    })

    for i in range(len(test_id)):
        preds.append({
            
        })

mae = np.array(mae_list)
rmse = np.array(rmse_list)
r2 = np.array(r2_list)

print("=== RandomForest Model Performance (subject-level generalization) ===")
print("Mean MAE per output:", mae.mean(axis=0))
print("Overall Mean MAE:", mae.mean())
print("Mean RMSE per output:", rmse.mean(axis=0))
print("Overall Mean RMSE:", rmse.mean())
print("Mean R2 per output:", r2.mean(axis=0))
print("Overall Mean R2:", r2.mean())
print("Best parameters:", best_params)