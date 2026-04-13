from data.dataset import generate_dataset, normalize
from model.activations import Linear, ReLU
from model.layers import Dense
from model.loss import mse
from model.mlp import MLP
from utils.visualization import plot_ground_truth

def main():
    print("=========================================")
    print("       MLP Function Approximation        ")
    print("=========================================")

    print("[1/6] Generating Raw Data...")
    X_raw, y_raw = generate_dataset(n_samples=2000)
    
    print("[2/6] Normalizing Dataset (Z-Score)...")
    X_train, y_train, stats = normalize(X_raw, y_raw)
    
    # Check if mean is ~0 and std is ~1
    print(f"      -> X_train Mean: {X_train.mean(axis=0)}") 
    print(f"      -> X_train Std:  {X_train.std(axis=0)}")
    
    print("[3/6] Visualizing Ground Truth...")
    
    # Choose your mode: "scatter" or "heatmap"
    plot_ground_truth(X_raw, y_raw, mode="scatter")

    print("\n--- Étape 1 Complete: Data is ready for the MLP ---")

    print("\n[4/6] Building MLP Model...")
    layers = [
            Dense(2, 64),   # first layer
            ReLU(),         # activation after first layer
            Dense(64, 64),
            ReLU(),
            Dense(64, 1),
            Linear()
        ]
    model = MLP(layers)

    print("\n[5/6] Performing Forward Pass...")
    # Forward pass
    y_pred = model.forward(X_train)

    print("[6/6] Computing Loss...")
    # Compute loss
    loss = mse(y_train, y_pred)

    print("Prediction shape:", y_pred.shape)
    print("Loss:", loss)

    print("\n--- Étape 2 Complete: Forward pass and loss computed ---")

if __name__ == "__main__":
    main()