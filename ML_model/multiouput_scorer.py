# define a scorer for multiple outputs
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def rmse_multioutput(y_true, y_pred):
    rmse_per_output = root_mean_squared_error(y_true, y_pred, multioutput='raw_values')
    return rmse_per_output.mean()