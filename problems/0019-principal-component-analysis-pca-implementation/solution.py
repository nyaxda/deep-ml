import numpy as np
from sklearn.preprocessing import StandardScaler

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    cov_matrix = np.cov(data_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    top_n = eigenvectors[:,-k:][:,::-1]
    for j in range(top_n.shape[1]):
        col = top_n[:, j]
        for val in col:
            if not np.isclose(val, 0):
                if val < 0:
                    top_n[:, j] = - col
                break
    return np.round(top_n, 4)
