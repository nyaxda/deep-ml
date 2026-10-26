from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	scaler = StandardScaler()
	mmaxscaler = MinMaxScaler()
	standardized_data = np.round(scaler.fit_transform(data),4).tolist()
	normalized_data = np.round(mmaxscaler.fit_transform(data), 4).tolist()
	return standardized_data, normalized_data