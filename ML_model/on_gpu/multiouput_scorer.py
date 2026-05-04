# define a scorer for multiple outputs
from sklearn.metrics import make_scorer
import cupy as cp
import numpy as np

def rmse_multioutput(y_true, y_pred):
    y_true = cp.asarray(y_true)
    y_pred = cp.asarray(y_pred)
    return -float(cp.sqrt(cp.mean((y_true - y_pred) ** 2)).get())

