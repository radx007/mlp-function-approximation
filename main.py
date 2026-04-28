from data.dataset import f, generate_dataset, normalize
from model.activations import Linear, ReLU, Tanh
from model.layers import Dense
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
    
    print("Normalizing Dataset (Z-Score)...")
    X_train, y_train, stats = normalize(X_raw, y_raw)
    
    print("Visualizing Ground Truth...")
    
    plot_ground_truth(X_raw, y_raw, mode="scatter")

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

    optimizer = SGD(model, lr=0.05)
    trainer = Trainer(model, optimizer)

    print("Starting Training...")
    trainer.train(X_train, y_train, epochs=2500,plot_fn=plot_loss)
    print("Training Completed!")

    plot_comparison(model, f, stats)
    
    log_model(model)


if __name__ == "__main__":
    main()