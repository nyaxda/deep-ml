import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    #computing auxiliary values
    y = np.array([A[1,0] + A[0,1], A[1,0] - A[0,1]])
    X = np.array([A[0,0] - A[1,1], A[0,0] + A[1,1]])

    #computing norms
    h = np.sqrt([y[0] ** 2 + X[0] ** 2, y[1] ** 2 + X[1] ** 2])

    #computing singular values
    s = np.array([(h[0] + h[1]) / 2, np.abs(h[0] - h[1]) / 2])

    #Building U
    t1 = X[0] / h[0]
    t2 = X[1] / h[1]
    cosine1 = (1 + t1)
    cosine2 = (1 + t2)
    sine1 = (1 - t1)
    sine2 = (1 - t2)
    cc = np.sqrt(cosine1 * cosine2)
    ss = np.sqrt(sine1 * sine2)
    cs = np.sqrt(cosine1 * sine2)
    sc = np.sqrt(sine1 * cosine2)

    c1 = (cc - ss) / 2
    s1 = (sc + cs) / 2

    U = np.array([[c1, -s1], [s1, c1]])

    V = np.diag(1 / s) @ U.T @ A
    return U, s, V