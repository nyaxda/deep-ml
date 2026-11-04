import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	n = features.shape[0]
	updated_weights = initial_weights.copy()
	updated_bias = initial_bias
	mse_values = []
	for _ in range(epochs):
		z = features @ updated_weights + updated_bias
		preds = 1 / (1 + np.exp(-z))
		mse_values.append(np.mean((preds - labels)**2))
		grad_w = 2/n * (preds - labels) * (preds * (1 - preds)) @ features
		grad_b = 2/n * np.sum((preds - labels) * (preds * (1 - preds)))

		updated_weights = updated_weights - learning_rate * grad_w
		updated_bias = updated_bias - learning_rate * grad_b

	return np.round(updated_weights, 4).tolist(), round(updated_bias, 4), np.round(mse_values, 4).tolist()