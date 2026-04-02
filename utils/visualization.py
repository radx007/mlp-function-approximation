from matplotlib import pyplot as plt

def plot_ground_truth(X, y, mode="heatmap"):
    # Use a consistent color map for all plots
    cmap = 'viridis'

    if mode == "heatmap":
        plt.figure(figsize=(8, 6))
        plt.tricontourf(X[:, 0], X[:, 1], y.flatten(), levels=20, cmap=cmap)
        plt.colorbar(label="Altitude (Z)")
        plt.title("Ground Truth: heatmap")
        plt.show()

    elif mode == "scatter":
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(X[:, 0], X[:, 1], y.flatten(), c=y.flatten(), cmap=cmap, s=15)
        fig.colorbar(img, ax=ax, label="Altitude")
        ax.set_title("Ground Truth: 3D Scatter Points")
        plt.show()
