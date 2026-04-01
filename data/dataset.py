import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)