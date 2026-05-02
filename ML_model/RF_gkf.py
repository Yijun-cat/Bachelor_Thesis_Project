# RF, Within-run temporal generalization

from build_dataframe import construct_df
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from rf_grid import rf_grid
from train_model import train_model

subjects = ['03', '04', '05', '06', '08', 11, 12,
            13, 15, 16, 17, 18, 19, 20, 22, 23, 24,
            27, 31, 32, 33, 35, 37, 38, 43
]

df_model, feature_cols, target_cols = construct_df(subjects)
# add boolean tag for trainval, test set
trainval_mask = ~df_model['is_temporal_test']
test_mask = df_model['is_temporal_test']
# build trainval, test sets
X_trainval = df_model.loc[trainval_mask, feature_cols].to_numpy()
y_trainval = df_model.loc[trainval_mask, target_cols].to_numpy()
X_test = df_model.loc[test_mask, feature_cols].to_numpy()
y_test = df_model.loc[test_mask, feature_cols].to_numpy()

base_model, param_grid, cv = rf_grid(df_model, X_trainval, y_trainval, cv_method='gkf')
best_model, best_params = train_model(X_trainval, y_trainval, base_model, param_grid, cv_method='gkf')

# Evaluate model on test set
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("===RandomForest Model Performance (within-run temporal generalization)===")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)