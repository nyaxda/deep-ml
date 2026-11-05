import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	batch = []
	for i in range(0, len(X), batch_size):
		batch_i_x = X[i : i + batch_size].tolist()
		if y is None:
			batch.append([batch_i_x])
		else:
			batch_i_y = y[i : i + batch_size].tolist()
			batch.append([batch_i_x, batch_i_y])			
	return batch
