import numpy as np


class ReLU:
    def forward(self, Z):
        self.Z = Z # store for future backprop
        return np.maximum(0, Z) 


class Linear:
    def forward(self, Z):
        return Z 