from matplotlib import pyplot as plt
import numpy as np

from model.loss import mse, mse_grad


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, X, y, epochs=1000, batch_size=32, plot_fn=None):
        losses = []
        ax = None
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.model.forward(X_batch)
                loss = mse(y_batch, y_pred)
                epoch_losses.append(loss)

                grad = mse_grad(y_batch, y_pred)
                self.model.backward(grad)
                self.optimizer.step()

            epoch_loss = float(np.mean(epoch_losses))
            losses.append(epoch_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")

            if plot_fn and epoch % 20 == 0:
                ax = plot_fn(losses, ax, live=True)

        if plot_fn:
            plt.ioff()
            plt.show()
        return losses