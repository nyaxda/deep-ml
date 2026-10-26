import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X_array = np.array(X)
	y_array = np.array(y)
	theta = np.linalg.inv(X_array.T @ X_array) @ X_array.T @ y
	theta = np.round(theta, 4).tolist()
	# Your code here, make sure to round
	return theta