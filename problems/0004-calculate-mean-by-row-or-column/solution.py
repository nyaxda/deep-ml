def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	import numpy as np
	matrix_array = np.array(matrix)
	means = np.mean(matrix_array, axis=0).tolist() if mode == "column" else np.mean(matrix_array, axis=1).tolist() if mode == "row" else []
	return means