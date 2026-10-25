def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	import numpy as np
	matrix_array = np.array(matrix)
	eigenvalues = np.linalg.eigvals(matrix_array)
	return eigenvalues.tolist()