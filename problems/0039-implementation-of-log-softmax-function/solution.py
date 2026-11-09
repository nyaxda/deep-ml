import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	scores = np.array(scores)
	maxx = np.max(scores)
	shifted = scores - maxx
	total = np.sum(np.exp(scores - maxx))
	return shifted - np.log(total)
