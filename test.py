import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def tanh_normalize(data):
	max_for_each_column = np.max(data, axis=0)
	min_for_each_column = np.min(data, axis=0)
	c = (max_for_each_column + min_for_each_column)/2
	for i in range(data.shape[1]):
		if max_for_each_column[i] != min_for_each_column[i]:
			data[:, i] = 1/2*np.tanh(4*(data[:, i] - c[i])/(max_for_each_column[i] - min_for_each_column[i]))+1/2
		else:
			data[:, i] = [0]*data.shape[0]
	return data


def normalize(data):
	scaler = MinMaxScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	return data


data = np.loadtxt('/Users/clas/Downloads/Work/Code/CognitiveStatusResearch/Dat/jm2/FlightParameter/1.csv', delimiter=',', skiprows=1, encoding='gb2312')
data1 = tanh_normalize(data)
data2 = normalize(data)
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(data1)
plt.title('tanhScaler')
plt.subplot(3, 1, 2)
plt.plot(data2)
plt.title('MinMaxScaler')
plt.subplot(3, 1, 3)
plt.plot(data2-data1)
plt.title('MinMaxScaler-tanhScaler')
plt.tight_layout()
plt.savefig('Results/normed_data.pdf', format='pdf')