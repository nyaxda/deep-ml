import numpy as np
import math

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []
	# Your code here
	for _ in range(n_clf):
		min_error = float('inf')
		best = {}
		best_preds = None
		for i in range(n_features):
			col = X[:, i]
			for t in np.unique(col):
				p = 1
				preds = np.where(col >= t, 1, -1)

				error = np.sum(w[preds != y])
				if error > 0.5:
					p = -1
					error = 1 - error

				if error < min_error:
					min_error = error
					best = {'polarity': p, 'threshold': t, 'feature_index': i}
					best_preds = p * preds
		
		alpha = 0.5 * np.log((1 - min_error)/ (min_error + 1e-10))

		w = w * np.exp(-alpha * y * best_preds)
		w = w / np.sum(w)

		best['alpha'] = alpha

		clfs.append(best)			

	return clfs
    