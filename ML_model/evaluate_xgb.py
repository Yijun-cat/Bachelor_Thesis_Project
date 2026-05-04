# functions evaluate XGBoost model performance
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

    X = df_model[feature_cols].to_numpy()
    y = df_model[target_cols].to_numpy()

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
        y_pred = best_model.predict(X_test)

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

    # Refit final model on all data for importance and SHAP
    best_model.fit(X, y)

    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # save outputs to csv
    if lag_feature == True:
        df_subject_metrics.to_csv(os.path.join(out_dir, "xgb_subject_metrics_lag.csv"), index=False)
        df_summary_full.to_csv(os.path.join(out_dir, "xgb_subject_summary_lag.csv"), index=False)
        df_predictions.to_csv(os.path.join(out_dir, "xgb_subject_predictions_lag.csv"), index=False)
        df_importance.to_csv(os.path.join(out_dir, "xgb_feature_importance_lag.csv"), index=False)
    else:
        df_subject_metrics.to_csv(os.path.join(out_dir, "xgb_subject_metrics.csv"), index=False)
        df_summary_full.to_csv(os.path.join(out_dir, "xgb_subject_summary.csv"), index=False)
        df_predictions.to_csv(os.path.join(out_dir, "xgb_subject_predictions.csv"), index=False)
        df_importance.to_csv(os.path.join(out_dir, "xgb_feature_importance.csv"), index=False)

    # Figure 1: RMSE by subject for one key output 
    key_output = shap_output_name if shap_output_name in target_cols else target_cols[-1]
    key_rmse_col = f"RMSE_{key_output}"

    if key_rmse_col in df_subject_metrics.columns:
        plt.figure(figsize=(10, 5))
        plot_df = df_subject_metrics.sort_values(key_rmse_col)
        sns.barplot(data=plot_df, x="sub_id", y=key_rmse_col, color="steelblue")
        plt.axhline(plot_df[key_rmse_col].mean(), color="red", linestyle="--", label="Mean") # add horizontal line at mean RMSE
        plt.xlabel("Held-out subject")
        plt.ylabel(f"RMSE of {key_output}")
        plt.title(f"Subject-level generalization: {key_output} RMSE by subject")
        plt.legend()
        plt.tight_layout()
        if lag_feature == True:
            plt.savefig(os.path.join(out_dir, f"rmse_by_subject_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, f"rmse_by_subject_{key_output}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Figure 2: True vs predicted scatter
    plt.figure(figsize=(6, 6))
    x_true = df_predictions[f"true_{key_output}"]
    y_pred_plot = df_predictions[f"pred_{key_output}"]
    plt.scatter(x_true, y_pred_plot, alpha=0.25)
    lims = [min(x_true.min(), y_pred_plot.min()), max(x_true.max(), y_pred_plot.max())]
    plt.plot(lims, lims, "r--") # draw line for perfect predictions
    plt.xlabel(f"True {key_output}")
    plt.ylabel(f"Predicted {key_output}")
    plt.title(f"True vs predicted {key_output}")
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"scatter_true_vs_pred_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"scatter_true_vs_pred_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: Example time-series trace
    example_sub = df_predictions["sub_id"].iloc[0]
    example_df = df_predictions[df_predictions["sub_id"] == example_sub].sort_values("time_s").copy()
    plt.figure(figsize=(12, 4))
    plt.plot(example_df["time_s"], example_df[f"true_{key_output}"], label=f"True {key_output}") # plot true time series
    plt.plot(example_df["time_s"], example_df[f"pred_{key_output}"], label=f"Predicted {key_output}", alpha=0.85) # plot predicted 
    plt.xlabel("Time (s)")
    plt.ylabel(key_output)
    plt.title(f"Example prediction trace for held-out subject {example_sub}")
    plt.legend()
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"timeseries_example_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"timeseries_example_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: RF model feature importance
    topn = min(10, len(df_importance)) # at least 10 features to plot
    df_top = df_importance.head(topn).sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(df_top["feature"], df_top["importance"], color="darkgreen") # plot horizontal bar chart
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top XGB feature importances")
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, "xgb_feature_importance_top10_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "xgb_feature_importance_top10.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP 
    shap_output_idx = target_cols.index(key_output)

    X_shap_df = df_model[feature_cols].copy()
    if len(X_shap_df) > shap_sample_size:
        X_shap_df = X_shap_df.sample(n=shap_sample_size, random_state=42)

    explainer = shap.TreeExplainer(best_model)
    shap_values_all = explainer.shap_values(X_shap_df)

    shap_values_target = extract_shap_for_output(shap_values_all, shap_output_idx)
    expected_value_target = extract_expected_value(explainer.expected_value, shap_output_idx)

    # SHAP summary beeswarm
    shap.summary_plot(
        shap_values_target,
        X_shap_df,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"shap_summary_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"shap_summary_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP summary bar
    shap.summary_plot(
        shap_values_target,
        X_shap_df,
        feature_names=feature_cols,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"shap_bar_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"shap_bar_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP local explanation
    shap_exp = shap.Explanation(
        values=shap_values_target[0], # a vector of SHAP values for one sample
        base_values=expected_value_target,
        data=X_shap_df.iloc[0].values,
        feature_names=feature_cols
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"shap_waterfall_example_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"shap_waterfall_example_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "df_subject_metrics": df_subject_metrics,
        "df_summary": df_summary_full,
        "df_predictions": df_predictions,
        "df_importance": df_importance,
        "shap_output": key_output,
        "out_dir": out_dir
    }

# Model performance evaluation on within-run temporal generalization
def evaluate_temporal_xgb(
    df_model,
    feature_cols,
    target_cols,
    best_model,
    lag_feature = False,
    out_dir="xgb_temporal_results",
    shap_output_name="total_error",
    shap_sample_size=1000
):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_style("whitegrid")

    trainval_mask = ~df_model["is_temporal_test"]
    test_mask = df_model["is_temporal_test"]
    
    # train, validation, test set separation
    X_trainval = df_model.loc[trainval_mask, feature_cols].to_numpy()
    y_trainval = df_model.loc[trainval_mask, target_cols].to_numpy()
    X_test = df_model.loc[test_mask, feature_cols].to_numpy()
    y_test = df_model.loc[test_mask, target_cols].to_numpy()

    df_test = df_model.loc[test_mask, ["sub_id", "run_id", "time_s", "level"]].copy()
    df_test["group_run"] = (df_test["sub_id"].astype(str) + "_" + df_test["run_id"].astype(str))

    # Fit tuned model on full trainval set
    best_model.fit(X_trainval, y_trainval)
    y_pred = best_model.predict(X_test)

    # performance metrics
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    rmse = root_mean_squared_error(y_test, y_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")

    df_summary = pd.DataFrame({
        "output": target_cols,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    overall_row = pd.DataFrame([{
        "output": "overall_mean_across_outputs",
        "MAE": np.mean(mae),
        "RMSE": np.mean(rmse),
        "R2": np.mean(r2)
    }])

    df_summary_full = pd.concat([df_summary, overall_row], ignore_index=True)

    # Prediction output
    df_predictions = df_test.copy()
    for i, col in enumerate(target_cols):
        df_predictions[f"true_{col}"] = y_test[:, i]
        df_predictions[f"pred_{col}"] = y_pred[:, i]

    # metrics for subjects' each run
    rows_run_metrics = []
    for run_name, df_run in df_predictions.groupby("group_run"):
        row = {
            "group_run": run_name,
            "sub_id": df_run["sub_id"].iloc[0],
            "run_id": df_run["run_id"].iloc[0],
        }

        for col in target_cols:
            y_true_run = df_run[f"true_{col}"].to_numpy()
            y_pred_run = df_run[f"pred_{col}"].to_numpy()

            row[f"MAE_{col}"] = mean_absolute_error(y_true_run, y_pred_run)
            row[f"RMSE_{col}"] = root_mean_squared_error(y_true_run, y_pred_run)
            row[f"R2_{col}"] = r2_score(y_true_run, y_pred_run)

        rows_run_metrics.append(row)

    df_run_metrics = pd.DataFrame(rows_run_metrics)
        
    # feature importance
    df_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # Save tables
    if lag_feature == True:
        df_summary_full.to_csv(os.path.join(out_dir, "xgb_temporal_summary_lag.csv"), index=False)
        df_run_metrics.to_csv(os.path.join(out_dir, "xgb_temporal_run_metrics_lag.csv"), index=False)
        df_predictions.to_csv(os.path.join(out_dir, "xgb_temporal_predictions_lag.csv"), index=False)
        df_importance.to_csv(os.path.join(out_dir, "xgb_temporal_feature_importance_lag.csv"), index=False)
    else:
        df_summary_full.to_csv(os.path.join(out_dir, "xgb_temporal_summary.csv"), index=False)
        df_run_metrics.to_csv(os.path.join(out_dir, "xgb_temporal_run_metrics.csv"), index=False)
        df_predictions.to_csv(os.path.join(out_dir, "xgb_temporal_predictions.csv"), index=False)
        df_importance.to_csv(os.path.join(out_dir, "xgb_temporal_feature_importance.csv"), index=False)

    # Main output to plot
    key_output = shap_output_name if shap_output_name in target_cols else target_cols[-1]

    # figure 1: RMSE by run
    key_rmse_col = f"RMSE_{key_output}"
    if key_rmse_col in df_run_metrics.columns:
        plot_df = df_run_metrics.sort_values(key_rmse_col).copy()

        plt.figure(figsize=(12, 5))
        sns.barplot(data=plot_df, x="group_run", y=key_rmse_col, color="steelblue")
        plt.axhline(plot_df[key_rmse_col].mean(), color="red", linestyle="--", label="Mean")
        plt.xlabel("Run")
        plt.ylabel(f"RMSE of {key_output}")
        plt.title(f"Temporal generalization: {key_output} RMSE by run")
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        if lag_feature == True:
            plt.savefig(os.path.join(out_dir, f"rmse_by_run_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(out_dir, f"rmse_by_run_{key_output}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Figure 2: True vs predicted scatter plot
    plt.figure(figsize=(6, 6))
    x_true = df_predictions[f"true_{key_output}"]
    y_pred_plot = df_predictions[f"pred_{key_output}"]
    plt.scatter(x_true, y_pred_plot, alpha=0.25)

    lims = [min(x_true.min(), y_pred_plot.min()), max(x_true.max(), y_pred_plot.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel(f"True {key_output}")
    plt.ylabel(f"Predicted {key_output}")
    plt.title(f"True vs predicted {key_output}")
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"scatter_true_vs_pred_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"scatter_true_vs_pred_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # figure 3: Example run trace
    example_run = df_predictions["group_run"].iloc[0]
    example_df = df_predictions[df_predictions["group_run"] == example_run].sort_values("time_s").copy()

    plt.figure(figsize=(12, 4))
    plt.plot(example_df["time_s"], example_df[f"true_{key_output}"], label=f"True {key_output}")
    plt.plot(example_df["time_s"], example_df[f"pred_{key_output}"], label=f"Predicted {key_output}", alpha=0.85)
    plt.xlabel("Time (s)")
    plt.ylabel(key_output)
    plt.title(f"Example temporal prediction trace for run {example_run}")
    plt.legend()
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"timeseries_example_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"timeseries_example_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # figure 4: Feature importance
    topn = min(10, len(df_importance))
    df_top = df_importance.head(topn).sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(df_top["feature"], df_top["importance"], color="darkgreen")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top XGB feature importances")
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, "xgb_feature_importance_top10_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "xgb_feature_importance_top10.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP
    shap_output_idx = target_cols.index(key_output)

    X_shap_df = df_model.loc[trainval_mask, feature_cols].copy()
    if len(X_shap_df) > shap_sample_size:
        X_shap_df = X_shap_df.sample(n=shap_sample_size, random_state=42)

    explainer = shap.TreeExplainer(best_model)
    shap_values_all = explainer.shap_values(X_shap_df)

    shap_values_target = extract_shap_for_output(shap_values_all, shap_output_idx)
    expected_value_target = extract_expected_value(explainer.expected_value, shap_output_idx)

    shap.summary_plot(
        shap_values_target,
        X_shap_df,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"shap_summary_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"shap_summary_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values_target,
        X_shap_df,
        feature_names=feature_cols,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"shap_bar_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    shap_exp = shap.Explanation(
        values=shap_values_target[0],
        base_values=expected_value_target,
        data=X_shap_df.iloc[0].values,
        feature_names=feature_cols
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    if lag_feature == True:
        plt.savefig(os.path.join(out_dir, f"shap_waterfall_example_{key_output}_lag.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"shap_waterfall_example_{key_output}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "df_summary": df_summary_full,
        "df_run_metrics": df_run_metrics,
        "df_predictions": df_predictions,
        "df_importance": df_importance,
        "shap_output": key_output,
        "out_dir": out_dir
    }