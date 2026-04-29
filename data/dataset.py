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


def create_mesh(x_range=(-5, 5), y_range=(-5, 5), resolution=100):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    return np.meshgrid(x, y)


def predict_on_grid(model, X_grid, Y_grid, stats=None):
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

    # Normalize
    if stats is not None:
        grid_points = (grid_points - stats["X_mean"]) / stats["X_std"]

    Z_pred = model.predict(grid_points)

    # Denormalize
    if stats is not None:
        Z_pred = Z_pred * stats["y_std"] + stats["y_mean"]

    return Z_pred.reshape(X_grid.shape)


def split_dataset(X, y, train_ratio=0.8, seed=42):
    np.random.seed(seed)

    indices = np.random.permutation(len(X))
    split_index = int(len(X) * train_ratio)

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test