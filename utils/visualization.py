import matplotlib.pyplot as plt
import numpy as np

from data.dataset import create_mesh, predict_on_grid


def plot_ground_truth(X, y, mode="heatmap"):
    # Use a consistent color map for all plots
    cmap = 'viridis'

    if mode == "heatmap":
        plt.figure(figsize=(8, 6))
        plt.tricontourf(X[:, 0], X[:, 1], y.flatten(), levels=20, cmap=cmap)
        plt.colorbar(label="Altitude (Z)")
        plt.title("Ground Truth: Heatmap")

    elif mode == "scatter":
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        img = ax.scatter(X[:, 0], X[:, 1], y.flatten(), c=y.flatten(), cmap=cmap, s=15)
        fig.colorbar(img, ax=ax, label="Altitude")
        ax.set_title("Ground Truth: 3D Scatter")

    plt.show()


def plot_loss(losses, ax=None, live=False):
    if live and ax is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.clear()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epochs")
    ax.grid(True)

    if live:
        plt.pause(0.01)
    else:
        plt.show()

    return ax


def plot_comparison(model, func, stats=None, resolution=100):
    X_grid, Y_grid = create_mesh(resolution=resolution)

    Z_true = func(X_grid, Y_grid)
    Z_pred = predict_on_grid(model, X_grid, Y_grid, stats)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X_grid, Y_grid, Z_true)
    ax1.set_title("Ground Truth")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X_grid, Y_grid, Z_pred)
    ax2.set_title("Model Prediction")

    plt.show()