import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) # Mean Squared Error

def mse_grad(y_true, y_pred):
    n = y_true.shape[0]
    return (2 / n) * (y_pred - y_true)