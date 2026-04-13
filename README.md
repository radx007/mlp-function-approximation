# MLP Function Approximation

## Step 1: Dataset Generation and Visualization

### What we did

* Generated 2000 random points (x, y) in the range [-5, 5]
* Computed the real output z using the function f(x, y) 
* Normalized the input (X) and output (y) using  Z-Score Normalization
* Visualized the ground truth using:
  * Heatmap
  * Surface plot

### Goal

Prepare clean data and understand the function before training the MLP

---

## Step 2: MLP Architecture and Forward Pass

### What we did

* Built a Multilayer Perceptron (MLP) from scratch using NumPy
* Defined the architecture: [2 → 64 → 64 → 1]
* Initialized weights using He initialization
* Implemented activation functions:

  * ReLU for hidden layers
  * Linear for output layer
* Implemented the forward pass through all layers
* Implemented Mean Squared Error (MSE) as the loss function
* Tested the model by generating predictions and computing the loss

### Goal

Build the neural network structure and ensure it can produce predictions before training
