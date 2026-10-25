import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	import numpy as np
	A_array = np.array(A)
	T_array = np.array(T)
	S_array = np.array(S)
	if (np.linalg.det(T_array) == 0 or np.linalg.det(S_array) == 0):
		return -1
	transformed_matrix = np.linalg.inv(T_array) @ A_array @ S_array
	return transformed_matrix