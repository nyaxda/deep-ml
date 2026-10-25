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
    # STEP 1: Build the symmetric matrix B = AᵀA.
    # A can be any shape and has no eigenvalues of its own, but AᵀA is always
    # square and symmetric, so it has real eigenvalues (these will be σ²)
    # and perpendicular eigenvectors (these will be the columns of V).
    B = A.T @ A

    # STEP 2: Find the rotation angle theta that zeroes out B's off-diagonal entry.
    # The condition for B[0,1] to become 0 after rotating is
    #     tan(2θ) = 2·B[0,1] / (B[0,0] - B[1,1])
    # so θ is half the arctan of that ratio.
    if B[0,0] == B[1,1]:
        # The ratio above would divide by zero (the tangent is infinite),
        # so 2θ = 90° and therefore θ = π/4 (45°). This is the special case
        # from the tutorial formula.
        theta = np.pi/4
    else:
        theta = 0.5 * np.arctan((2*B[0,1]/(B[0,0] - B[1,1])))

    # STEP 3: Build the 2x2 rotation matrix R from theta.
    # Angles are in radians; c and s are just two numbers placed in a grid.
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # STEP 4: Apply the rotation: D = RᵀBR.
    # D is (almost) diagonal. The off-diagonals are ~0 (tiny float leftovers
    # like 1e-16 count as zero), and the diagonal holds the eigenvalues of B,
    # which equal the squared singular values σ².
    D = R.T @ B @ R

    # STEP 5: Get the singular values.
    # np.diag(D) extracts the diagonal as a flat array (2D in, 1D out),
    # and sqrt turns each σ² into σ.
    S = np.sqrt(np.diag(D))

    # STEP 6: Sort the singular values largest first (SVD convention).
    # np.argsort returns POSITIONS that would sort S smallest first, not values,
    # and [::-1] flips that to largest first. We keep `order` in its own
    # variable because V needs the exact same reordering in step 8.
    order = np.argsort(S)
    order = order[::-1]
    S = S[order]

    # STEP 7: Guard against a zero singular value.
    # Step 9 needs Σ⁻¹, and a matrix with a zero on its diagonal has no inverse
    # (the original A would be singular). Returning -1 is the signal for that.
    # Note: S == 0 compares floats exactly, so a value like 1e-17 slips through.
    # np.isclose(S, 0) would be the more robust check.
    if np.any(S == 0):
        return -1

    # Turn the flat array S into the diagonal matrix Σ (1D in, 2D out).
    Sigma = np.diag(S)

    # STEP 8: Build V from the columns of R (the eigenvectors of AᵀA),
    # reordered to match S.
    # Each singular value has a partner column in V, so when S was reordered,
    # the columns must move with it. V[:, order] means "all rows, columns
    # in this order". Using the same `order` as S keeps the pairs intact.
    V = R
    V = V[:, order]

    # STEP 9: Recover U from A = UΣVᵀ.
    # Multiply both sides by V on the right (VᵀV = I):   A·V = U·Σ
    # Multiply both sides by Σ⁻¹ on the right:           U = A·V·Σ⁻¹
    U = A @ V @ np.linalg.inv(Sigma)

    # STEP 10: Return U, the sorted singular values, and Vᵀ.
    # The docstring asks for the transpose of V, so we return V.T.
    return U, S, V.T