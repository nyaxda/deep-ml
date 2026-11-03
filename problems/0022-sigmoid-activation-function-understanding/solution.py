import math

def sigmoid(z: float) -> float:
	#Your code 
	sigmoid = 1/(1 + math.exp(-z))
	return round(sigmoid, 4)