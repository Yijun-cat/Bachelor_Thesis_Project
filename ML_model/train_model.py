# function training model and selecet the best model using Grid Search Cross Validation
from sklearn.model_selection import GridSearchCV

# train model function
def train_model_rf(X_trainval, y_trainval, base_model, param_grid, cv):
    grid = GridSearchCV(
        estimator = base_model,
        param_grid = param_grid,
        scoring = 'neg_root_mean_squared_error',
        cv = cv,
        n_jobs = -1,
        verbose = 1
    )

    # fit train and validation set
    grid.fit(X_trainval, y_trainval)

    # get the best model and its parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params

def train_model_xgb(X_trainval, y_trainval, base_model, param_grid, cv):
    grid = GridSearchCV(
        estimator = base_model,
        param_grid = param_grid,
        scoring = 'neg_root_mean_squared_error',
        cv = cv,
        n_jobs = -1,
        verbose = 1
    )

    # fit train and validation set
    grid.fit(X_trainval, y_trainval)

    # get the best model and its parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params

def train_model_cat(X_trainval, y_trainval, base_model, param_grid, cv):
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=8,
        verbose=1
    )

    grid.fit(X_trainval, y_trainval)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    return best_model, best_params
