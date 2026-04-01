import numpy as np

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

def generate_dataset(n_samples=2000, seed=42):
    np.random.seed(seed)

    X = np.random.uniform(-5, 5, (n_samples, 2)) # 2000 examples(N, 2)

    z = f(X[:, 0], X[:, 1])  # f(x,y)

    y = z.reshape(-1, 1) # turn into (N, 1)

    return X, y 
