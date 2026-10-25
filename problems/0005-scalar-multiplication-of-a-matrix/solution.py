def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	result = [[j[i]*scalar for i in range(len(j))] for j in matrix]
	return result