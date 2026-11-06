import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	passed = []
	failed = []
	final = []
	for i in X:
		if i[feature_i] >= threshold:
			passed.append(i.tolist())
		else:
			failed.append(i.tolist())
	final.append(np.array(passed))
	final.append(np.array(failed))
	return final