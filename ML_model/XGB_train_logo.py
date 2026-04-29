
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from build_dataframe import construct_df


subjects = ['03', '04', '05', '06', '08', 11, 12,
            13, 15, 16, 17, 18, 19, 20, 22, 23, 24,
            27, 31, 32, 33, 35, 37, 38, 43
]

df_model, X, y = construct_df(subjects)

# XGB base model
xgb_base = make_pipeline(
    SimpleImputer(strategy='mean'),
    MultiOutputRegressor(
        XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,
            device = "cuda",
        )
    )
)

# possible parameters
param_grid_xgb = {
    'multioutputregressor__estimator__n_estimators': [25, 50, 100],
    "multioutputregressor__estimator__max_depth": [2, 3, 4, 5],
    "multioutputregressor__estimator__colsample_bytree": [0.25, 0.5],
    "multioutputregressor__estimator__booster": ["dart", "gbtree"]
}

# Leave one group out cross-validator
groups = df_model['sub_id']
logo = LeaveOneGroupOut()
cv = logo.split(X, y, groups)

