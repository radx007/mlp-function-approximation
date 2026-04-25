import numpy as np
 
class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim) # He initialization
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X  # store for future backprop
        Z = X @ self.W + self.b
        return Z
    
    def backward(self, dZ):

        # dW = Xᵀ · δ
        self.dW = self.X.T @ dZ

        # db = Σ δ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # dX = δ · Wᵀ
        dX = dZ @ self.W.T

        return dX