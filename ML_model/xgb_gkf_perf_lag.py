from build_dataframe import construct_df
from xgb_grid import xgb_grid
from train_model import train_model_xgb
from evaluate_xgb import evaluate_temporal_xgb

subjects = [
    "03", "04", "05", "06", "08",
    "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "22",
    "23", "24", "27", "31", "32",
    "33", "35", "37", "38", "43"
]

df_model, feature_cols, target_cols = construct_df(subjects, with_lag_feature=True)

trainval_mask = ~df_model["is_temporal_test"]
X_trainval = df_model.loc[trainval_mask, feature_cols].to_numpy()
y_trainval = df_model.loc[trainval_mask, target_cols].to_numpy()

base_model, param_grid, cv = xgb_grid(df_model, X_trainval, y_trainval, cv_method="gkf")
best_model, best_params = train_model_xgb(X_trainval, y_trainval, base_model, param_grid, cv)

results = evaluate_temporal_xgb(
    df_model=df_model,
    feature_cols=feature_cols,
    target_cols=target_cols,
    best_model=best_model,
    lag_feature=True,
    out_dir="xgb_temporal_results",
    shap_output_name="total_error",
    shap_sample_size=1000
)

print("=== XGBoost Performance with lag features (within-run temporal generalization) ===")
print("Best parameters:", best_params)
print(results["df_summary"])
print("Saved outputs to:", results["out_dir"])