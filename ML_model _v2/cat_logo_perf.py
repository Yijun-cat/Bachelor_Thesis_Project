# CatBoost model subject-level generalization without lagged features
from build_dataframe import construct_df
from cat_grid import cat_grid
from train_model import train_model_cat
from evaluate_cat import evaluate_subject_level_cat

subjects = [
    "03", "04", "05", "06", "08",
    "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "22",
    "23", "24", "27", "31", "32",
    "33", "35", "37", "38", "43"
]

df_model, feature_cols, target_cols = construct_df(subjects, with_lag_feature=False)
X = df_model[feature_cols].to_numpy()
y = df_model[target_cols].to_numpy()

base_model, param_grid, cv = cat_grid(df_model, X, y, cv_method="logo")
best_model, best_params = train_model_cat(X, y, base_model, param_grid, cv)

# evaluate model performance
results = evaluate_subject_level_cat(
    df_model=df_model,
    feature_cols=feature_cols,
    target_cols=target_cols,
    best_model=best_model,
    lag_feature=False,
    out_dir="cat_subject_results",
    shap_output_name="total_error"
)

print("=== Catboost Performance (subject-level generalization) ===")
print(results["df_summary"])
print("Best parameters:", best_params)