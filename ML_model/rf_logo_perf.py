# train and evaluate RandomForest model performance (subject-level generalization)
from build_dataframe import construct_df
from rf_grid import rf_grid
from train_model import train_model
from evaluate_rf import evaluate_subject_level_rf

subjects = [
    "03", "04", "05", "06", "08",
    "11", "12", "13", "15", "16",
    "17", "18", "19", "20", "22",
    "23", "24", "27", "31", "32",
    "33", "35", "37", "38", "43"
]
    
# get dataframe, features and targets columns
df_model, feature_cols, target_cols = construct_df(["03", "04", "05", "06", "08"])
X = df_model[feature_cols].to_numpy()
y = df_model[target_cols].to_numpy()

# Hyper parameter tuning using grid search
base_model, param_grid, cv = rf_grid(df_model, X, y, cv_method='logo')
best_model, best_params = train_model(X, y, base_model, param_grid, cv)

results = evaluate_subject_level_rf(
    df_model=df_model,
    X=X,
    y=y,
    feature_cols=feature_cols,
    target_cols=target_cols,
    best_model=best_model,
    out_dir="rf_subject_results",
    shap_output_name="total_error",
    shap_sample_size=1000
)

print(results["df_summary"])
print("Saved outputs to:", results["out_dir"])