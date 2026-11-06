import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	if n_col == None:
		n_col = max(x) + 1
	array = np.zeros((len(x), n_col))
	
	for i, value in enumerate(x):
		array[i, value] = 1
	return array