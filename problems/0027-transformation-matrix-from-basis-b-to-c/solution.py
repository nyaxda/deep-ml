def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	aug_mat = [C[i] + B[i] for i in range(len(C))]
	for j in range(len(C)):
		pivot = aug_mat[j][j]
		aug_mat[j] = [el / pivot for el in aug_mat[j]]
		for k in range(len(C)):
			if k != j:
				div_el = aug_mat[k][j]
				aug_mat[k] = [x - y for x, y in zip(aug_mat[k], [el * div_el for el in aug_mat[j]])]		
	P = [row[len(C):] for row in aug_mat]
	return P