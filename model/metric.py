from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def RMSE(target, output):
    return np.sqrt(mean_squared_error(target, output))

def MAE(target, output):
    return mean_absolute_error(target, output)

