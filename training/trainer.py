from matplotlib import pyplot as plt

from model.loss import mse, mse_grad


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, X, y, epochs=1000, plot_fn=None):
        losses = []
        ax = None

        for epoch in range(epochs):
            # Forward
            y_pred = self.model.forward(X)

            # Loss
            loss = mse(y, y_pred)
            losses.append(loss)

            # Backward
            grad = mse_grad(y, y_pred)
            self.model.backward(grad)

            # Update
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if plot_fn and epoch % 20 == 0:
                ax = plot_fn(losses, ax, live=True)

        if plot_fn:
            plt.ioff()
            plt.show()
        return losses