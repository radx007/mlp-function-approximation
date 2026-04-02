import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

def generate_dataset(n_samples=2000, seed=42):
    np.random.seed(seed)

    X = np.random.uniform(-5, 5, (n_samples, 2)) # 2000 examples(N, 2)

    z = f(X[:, 0], X[:, 1])  # f(x,y)

    y = z.reshape(-1, 1) # turn into (N, 1)

    return X, y 

def normalize(X, y):
    X_mean = X.mean(axis=0) # Get the mean (center) 
    X_std = X.std(axis=0) + 1e-8 # Get the standard deviation (spread) with epsilon to avoid division by zero

    y_mean = y.mean(axis=0) 
    y_std = y.std(axis=0) + 1e-8

    # z-score normalization
    X_norm = (X - X_mean) / X_std 
    y_norm = (y - y_mean) / y_std

    stats = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std
    }

    return X_norm, y_norm, stats