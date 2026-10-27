import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    fold_size_list = []
    fold_size = n_samples // k
    remainder = n_samples - (fold_size * k)
    iterations = k
    while iterations:
        fold_size_list.append(fold_size)
        iterations -= 1
        i = 0
    while remainder:
        fold_size_list[i] += 1
        remainder -= 1
        if i == len(fold_size_list) - 1:
            i = 0
        i+= 1
    start = 0
    folds = []
    for i in range(len(fold_size_list)):
        folds.append(indices[start:start + fold_size_list[i]])
        start += fold_size_list[i]
    results = []
    for i in range(len(folds)):
        test_data = folds[i].tolist()
        train_chunks = folds[:i] + folds[i+1:]
        train_data = [idx for chunk in train_chunks for idx in chunk]
        train, test = train_data, test_data
        pair = train, test
        results.append(pair)
    return results