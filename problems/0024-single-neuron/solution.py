import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	X = features
	y = labels
	probabilities = []
	for xi in X:
		z = sum(k * w for k, w in zip (xi, weights)) + bias
		probabilities.append(round(1 / (1 + math.exp(-z)), 4))
	n = len(X)
	mse = round(1/n * sum((p - t)**2 for p, t in zip(probabilities, y)), 4)
	return probabilities, mse