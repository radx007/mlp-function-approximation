import numpy as np


class ReLU:
    def forward(self, Z):
        self.Z = Z # store for future backprop
        return np.maximum(0, Z) 
    
    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ


class Linear:
    def forward(self, Z):
        return Z

    def backward(self, dA):
        return dA 
    
class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1 - self.A**2)
        return dZ    