from data.dataset import generate_dataset, normalize
from utils.visualization import plot_ground_truth

def main():
    print("=========================================")
    print("       MLP Function Approximation        ")
    print("=========================================")

    print("[1/3] Generating Raw Data...")
    X_raw, y_raw = generate_dataset(n_samples=2000)
    
    print("[2/3] Normalizing Dataset (Z-Score)...")
    X_train, y_train, stats = normalize(X_raw, y_raw)
    
    # Check if mean is ~0 and std is ~1
    print(f"      -> X_train Mean: {X_train.mean(axis=0)}") 
    print(f"      -> X_train Std:  {X_train.std(axis=0)}")
    
    print("[3/3] Visualizing Ground Truth...")
    
    # Choose your mode: "scatter" or "heatmap"
    plot_ground_truth(X_raw, y_raw, mode="scatter") 
    plot_ground_truth(X_raw, y_raw, mode="heatmap") 

    print("\n--- Étape 1 Complete: Data is ready for the MLP ---")

if __name__ == "__main__":
    main()