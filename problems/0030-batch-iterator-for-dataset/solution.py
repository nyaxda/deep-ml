import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	batch = []
	for i in range(0, len(X), batch_size):
		batch_i_x = X[i : i + batch_size].tolist()
		batch_i_y = y[i : i + batch_size].tolist() if y is not None else None
		if batch_i_y is not None:
			batch.append([batch_i_x, batch_i_y])
		else:
			batch.append([batch_i_x])
	return batch
