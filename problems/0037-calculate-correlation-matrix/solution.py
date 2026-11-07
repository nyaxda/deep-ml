import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	X_centered = X - X.mean(axis=0)
	n = X.shape[0]
	if Y is not None:
		Y_centered = Y- Y.mean(axis=0)
	else:
		Y = X
		Y_centered = X_centered
	cov_mat = X_centered.T @ Y_centered / (n - 1)
	std_X = np.std(X, axis=0, ddof=1)
	std_Y = np.std(Y, axis=0, ddof=1)

	corr_mat = cov_mat/ np.outer(std_X, std_Y)

	return corr_mat