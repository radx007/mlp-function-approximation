import numpy as np

def save_model_npz(model, filename="mlp_model.npz"):
    params = {}

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "W"):
            params[f"W_{i}"] = layer.W
            params[f"b_{i}"] = layer.b

    np.savez(filename, **params)

    print(f"Model saved to {filename}")


def load_model_npz(model, filename="mlp_model.npz"):
    data = np.load(filename)

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "W"):
            layer.W = data[f"W_{i}"]
            layer.b = data[f"b_{i}"]

    print(f"Model loaded from {filename}")