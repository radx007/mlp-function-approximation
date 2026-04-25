class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db