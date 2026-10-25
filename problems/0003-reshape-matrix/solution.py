import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	import numpy as np
	a_array = np.array(a)
	try:
		reshaped_matrix = a_array.reshape(new_shape).tolist()
	except ValueError:
		reshaped_matrix= []
	return reshaped_matrix