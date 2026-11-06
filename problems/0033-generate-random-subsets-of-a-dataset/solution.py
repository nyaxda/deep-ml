import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    subsets = []
    n = X.shape[0]
    for _ in range(n_subsets):
        if replacements == True:
            picks = np.random.choice(len(X), size=n, replace=True)
        else:
            picks = np.random.choice(len(X), size=n//2, replace=False)
        X_sample = X[picks]
        y_sample = y[picks]
        subsets.append((X_sample.tolist(), y_sample.tolist()))
    return subsets