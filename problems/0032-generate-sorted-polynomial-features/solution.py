import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    # ✏️  Your code here
    n_samples, n_features = X.shape
    combo_list = []
    for n in range(degree + 1):
        combo_list.extend(combinations_with_replacement(range(n_features), n))
    poly_features = []
    for combos in combo_list:
        if len(combos) == 0:
            col = np.ones((n_samples, 1))
        else:
            col = np.prod(X[:, combos], axis=1, keepdims=True)
        poly_features.append(col)
    X_poly_manual = np.hstack(poly_features)
    return np.sort(X_poly_manual, axis = 1)