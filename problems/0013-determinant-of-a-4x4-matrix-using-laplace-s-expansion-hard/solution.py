def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	det = 0
	for i in range(len(matrix[0])):
		remaining_rows = matrix[1:]
		minor_matrix = [row[:i] + row[i+1:] for row in remaining_rows]
		sign = (-1) ** (0 + i)
		if len(minor_matrix) == 2 and all(len(row) == 2 for row in minor_matrix):
			det += matrix[0][i] * sign * (minor_matrix[0][0] * minor_matrix[1][1] - minor_matrix[0][1] * minor_matrix[1][0])
		else:
			det += matrix[0][i] * sign * determinant_4x4(minor_matrix)
	return det