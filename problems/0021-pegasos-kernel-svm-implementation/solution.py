import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    X = data
    y = labels
    n = X.shape[0]
    alpha = np.zeros(n)
    b = 0.0

    for t in range(1, iterations + 1):
        nt = 1 / (lambda_val * t)
        for i in range(n):
            xi, yi = X[i], y[i]
            if kernel == 'linear':
                f = sum(alpha[j] * y[j] * np.dot(X[j], xi) for j in range(n)) + b
            elif kernel == 'rbf':
                f = sum(alpha[j] * y[j] * np.exp(-np.sum((np.square(np.abs(X[j] - xi)) / (2 * sigma ** 2)))) for j in range(n)) + b
            if (yi * f) < 1:
                alpha[i] = (1 - nt * lambda_val) * alpha[i] + nt
                b = b + nt * yi
            else:
                alpha[i] = (1 - nt * lambda_val) * alpha[i]

    return alpha, b