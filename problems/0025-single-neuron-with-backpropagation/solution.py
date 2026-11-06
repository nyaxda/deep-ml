import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []
    n = features.shape[0]

    for _ in range(epochs):
        # Forward pass
        z = features @ weights + bias
        sigma = 1 / (1 + np.exp(-z))

        # MSE loss
        mse = np.mean((sigma - labels) ** 2)
        mse_values.append(round(float(mse), 4))

        # Backward pass
        error = (sigma - labels) * sigma * (1 - sigma)
        grad_w = (2 / n) * (features.T @ error)
        grad_b = (2 / n) * np.sum(error)

        # Update
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return np.round(weights, 4), round(float(bias), 4), mse_values