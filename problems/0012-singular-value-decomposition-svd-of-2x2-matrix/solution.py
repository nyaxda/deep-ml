import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B = A.T @ A

    if B[0,0] == B[1,1]:
        theta = np.pi/4
    else:
        theta = 0.5 * np.arctan((2*B[0,1]/(B[0,0] - B[1,1])))
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    D = R.T @ B @ R
    S = np.sqrt(np.diag(D))
    order = np.argsort(S)
    order = order[::-1]
    S = S[order]
    if np.any(S == 0):
        return -1
    Sigma = np.diag(S)
    V = R
    V = V[:,order]
    U = A @ V @ np.linalg.inv(Sigma)

    return U, S, V.T