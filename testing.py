import numpy as np
import copy
import matplotlib.pyplot as plt
from get_data import tanh_normalize
from segment import *
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2



X = np.loadtxt('HAPT Data Set/Train/X_train.txt', delimiter=' ')
y = np.loadtxt('HAPT Data Set/Train/y_train.txt')

valid_index = []
for i, item in enumerate(y):
	if item < 7:
		valid_index.append(i)

X = X[valid_index]
y = y[valid_index]


Z = tanh_normalize(X)
X = Z[0]
par = Z[1]
selector = SelectKBest(chi2, k=20)
X = selector.fit_transform(X, y)
selected_index = selector.get_support(True)

print('Selected.')

oss = OnlineSmoothSegment(window_size=2, lambda_parameter=.11)
oss.fit(X, y)

# test_X = np.loadtxt('HAPT Data Set/Train/X_train.txt', delimiter=' ')[728:1051]
# test_y = np.loadtxt('HAPT Data Set/Train/y_train.txt')[728:1051]
test_X = np.loadtxt('HAPT Data Set/Test/X_test.txt', delimiter=' ')[:324]
test_y = np.loadtxt('HAPT Data Set/Test/y_test.txt')[:324]

test_X = tanh_normalize(test_X, par)[0]

valid_index = []
for i, item in enumerate(test_y):
	if item < 7:
		valid_index.append(i)

test_X = test_X[valid_index]
test_y = test_y[valid_index]

test_X = test_X[:, selected_index]

oss.predict(test_X, beta=0.01, init_cluster=5)

plt.plot(test_y, label='true')
plt.plot(oss.predict_result, label='predict')
plt.legend()

Acc = 0
for i in range(min(len(y), len(oss.predict_result))):
	if oss.predict_result[i] == test_y[i]:
		Acc += 1
plt.title('Accuracy: %.2f%%' % (Acc / len(test_y) * 100))
plt.tight_layout()
plt.show()
