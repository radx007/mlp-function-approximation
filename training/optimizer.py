import numpy as np
class SGD:
    def __init__(self, model, lr=0.15, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        
        # Initialize velocity for each layer
        self.velocities = []
        for layer in model.layers:
            if hasattr(layer, "W"):
                self.velocities.append({
                    "dW": np.zeros_like(layer.W),
                    "db": np.zeros_like(layer.b)
                })
            else:
                self.velocities.append(None)

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, "W"):
                v = self.velocities[layer_idx]
                
                # Update velocity
                v["dW"] = self.momentum * v["dW"] + (1 - self.momentum) * layer.dW
                v["db"] = self.momentum * v["db"] + (1 - self.momentum) * layer.db
                
                # Update weights with velocity
                layer.W -= self.lr * v["dW"]
                layer.b -= self.lr * v["db"]