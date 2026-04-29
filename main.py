from data.dataset import f, generate_dataset, normalize, split_dataset
from model.activations import Linear, ReLU, Tanh
from model.layers import Dense
from model.loss import mse
from model.mlp import MLP
from training.trainer import Trainer
from utils.log import log_model
from utils.visualization import plot_comparison, plot_ground_truth , plot_loss
from training.optimizer import SGD
import json
from datetime import datetime

def main():
    print("=========================================")
    print("       MLP Function Approximation        ")
    print("=========================================")

    print("Generating Raw Data...")
    X_raw, y_raw = generate_dataset(n_samples=2000)

    print("Splitting Dataset...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_dataset(X_raw, y_raw, train_ratio=0.8)

    
    print("Normalizing Dataset (Z-Score)...")
    X_train, y_train, stats = normalize(X_train_raw, y_train_raw)

    print("Normalizing Test Set with Training Stats...")
    X_test = (X_test_raw - stats["X_mean"]) / stats["X_std"]
    y_test = (y_test_raw - stats["y_mean"]) / stats["y_std"]
    
    print("Visualizing Ground Truth...")
    
    plot_ground_truth(X_raw, y_raw, mode="scatter") # select the mode you prefer: "heatmap" or "scatter"

    print("Building MLP Model...")
    layers = [
            Dense(2, 128),   # first layer
            Tanh(),          # activation after first layer
            Dense(128, 128),
            Tanh(), 
            Dense(128, 64),
            Tanh(),
            Dense(64, 1),
            Linear()
        ]
    model = MLP(layers)

    optimizer = SGD(model, lr=0.15, momentum=0.9)
    trainer = Trainer(model, optimizer)

    print("Starting Training...")
    trainer.train(X_train, y_train, epochs=1000, batch_size=32, plot_fn=plot_loss)
    print("Training Completed!")

    print("Evaluating on Test Set...")
    y_test_pred = model.predict(X_test)
    test_loss = mse(y_test, y_test_pred)
    print(f"Test MSE: {test_loss}")

    plot_comparison(model, f, stats)
    
    log_model(model)


if __name__ == "__main__":
    main()