# function training model and selecet the best model using Grid Search Cross Validation
import os
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import cupy as cp
from ML_model.on_gpu.multiouput_scorer import rmse_multioutput
from sklearn import set_config
set_config(array_api_dispatch=True)

# train model function
def train_model_gpu(X_trainval, y_trainval, base_model, param_grid, cv, groups=None, device="cpu"):
    # move trainval set to GPU
    scorer = make_scorer(rmse_multioutput, greater_is_better=False)

    grid = GridSearchCV(
        estimator = base_model,
        param_grid = param_grid,
        scoring = scorer if device =="cuda" else 'neg_root_mean_squared_error',
        cv = cv,
        n_jobs = 1 if device == "cuda" else -1,
        verbose = 1
    )

    # fit train and validation set
    if groups is not None:
        grid.fit(X_trainval, y_trainval, groups=groups)
    else:
        grid.fit(X_trainval, y_trainval)

    # get the best model and its parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params