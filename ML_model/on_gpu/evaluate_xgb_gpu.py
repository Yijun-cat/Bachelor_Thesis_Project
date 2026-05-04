# functions evaluate XGBoost model performance
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import cudf
import cupy as cp

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Helper function get shap values
def extract_shap_for_output(shap_values, output_idx):
    arr = np.array(shap_values, dtype=object) # make shap_values into numpy array

    if isinstance(shap_values, list): # shap returns a list, each element is 2D array (n_samples, n_features)
        return np.array(shap_values[output_idx]) # take the output_idx‑th element and convert it to a numpy array
    arr_num = np.array(shap_values) # other cases convert to a numeric array

    if arr_num.ndim == 3: # if shap returns 3D array
        # check shapes (n_outputs, n_samples, n_featuers) or (n_samples, n_features, n_outputs)
        if arr_num.shape[0] in [1, 2, 3, 4, 5, 6, 7, 8]: # if axis 0 is n_outputs
            return arr_num[output_idx]
        else: # if axis 2 is n_outputs
            return arr_num[:, :, output_idx]
        
    if arr_num.ndim == 2: # if shap_values is a 2D matrix (n_samples, n_features)
        return arr_num

    raise ValueError(f"Unsupported SHAP shape: {arr_num.shape}")

# helper function get expected values from shap
def extract_expected_value(expected_value, output_idx):
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        # convert to 1D array, flatten and choose scalar expected value for the output
        return np.array(expected_value).reshape(-1)[output_idx]
    # if expected_value is a single scalar
    return expected_value

# Model performance evaluation on subject-level generalizatioon
def evaluate_subject_level_xgb(
    df_model,
    feature_cols,
    target_cols,
    best_model,
    lag_feature = False,
    out_dir = "xgb_subject_results",
    shap_output_name = "total_error",
    shap_sample_size = 1000, # set shap sample size instead using whole dataset size
):
    os.makedirs(out_dir, exist_ok=True) # create a directory for saving outputs
    sns.set_style("whitegrid") # set white background with lgiht grid lines

    df_model = cudf.from_pandas(df_model)
    # get features and targets as cuDF DataFrames
    gdf_X = df_model[feature_cols]        
    gdf_y = df_model[target_cols]      
    # convert feature and targets to cupy array
    X = cp.array(gdf_X.to_cupy())
    y = cp.array(gdf_y.to_cupy())

    # list for storing outputs and performance metrics
    mae_list, rmse_list, r2_list = [], [], []
    rows_metrics = []
    rows_preds = []

    logo = LeaveOneGroupOut()
    groups = df_model["sub_id"].to_numpy() # group subject index

    # cross validation using best model on the whole dataset
    for train_id, test_id in logo.split(X, y, groups):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]
        subject_test = df_model.iloc[test_id]["sub_id"].iloc[0]

        best_model.fit(X_train, y_train)
        X_test = cp.asnumpy(X_test)
        y_test = cp.asnumpy(y_test)
        y_pred = best_model.predict(X_test)
        #y_pred = cp.asarray(best_model.predict(X_test))  
        # store performance per fold 
        mae_fold = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        rmse_fold = root_mean_squared_error(y_test, y_pred, multioutput="raw_values")
        r2_fold = r2_score(y_test, y_pred, multioutput="raw_values")
        mae_list.append(mae_fold)
        rmse_list.append(rmse_fold)
        r2_list.append(r2_fold)

        rows_metrics.append({
            "sub_id": subject_test,
            **{f"MAE_{target_cols[i]}": mae_fold[i] for i in range(len(target_cols))},
            **{f"RMSE_{target_cols[i]}": rmse_fold[i] for i in range(len(target_cols))},
            **{f"R2_{target_cols[i]}": r2_fold[i] for i in range(len(target_cols))},
        })

        df_fold = df_model.iloc[test_id][["sub_id", "time_s", "level", "run_id"]].copy()

        for i, col in enumerate(target_cols):
            df_fold[f"true_{col}"] = y_test[:, i]
            df_fold[f"pred_{col}"] = y_pred[:, i]

        rows_preds.append(df_fold)

    # convert list to array
    mae_arr = np.array(mae_list)
    rmse_arr = np.array(rmse_list)
    r2_arr = np.array(r2_list)

    # build dataframe for metrics per subject
    # all test predictions across all folds, one row per time point
    df_subject_metrics = pd.DataFrame(rows_metrics)
    df_predictions = pd.concat(rows_preds, axis=0, ignore_index=True)

    df_summary = pd.DataFrame({
        "output": target_cols,
        "MAE_mean": mae_arr.mean(axis=0),
        "MAE_std": mae_arr.std(axis=0),
        "RMSE_mean": rmse_arr.mean(axis=0),
        "RMSE_std": rmse_arr.std(axis=0),
        "R2_mean": r2_arr.mean(axis=0),
        "R2_std": r2_arr.std(axis=0),
    })

    overall_row = pd.DataFrame([{
        "output": "overall_mean_across_outputs",
        "MAE_mean": mae_arr.mean(),
        "MAE_std": np.nan,
        "RMSE_mean": rmse_arr.mean(),
        "RMSE_std": np.nan,
        "R2_mean": r2_arr.mean(),
        "R2_std": np.nan,
    }])

    df_summary_full = pd.concat([df_summary, overall_row], ignore_index=True)