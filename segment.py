import numpy as np
from get_data import Data, save_data, load_data
from admm import *
import copy
import matplotlib.pyplot as plt


class OnlineSmoothSegment:
	def __init__(self, window_size=1, lambda_parameter=0.11):
		self.window_size = window_size
		self.lambda_parameter = lambda_parameter
		self.input_size = 0
		
	def save_MRFs(self):
		save_data([self.tic, self.mean], 'Results/MRFs_w='+str(self.window_size)+'lambda='+str(self.lambda_parameter)+'.pkl')
		print('Results/MRFs_w='+str(self.window_size)+'lambda='+str(self.lambda_parameter)+'.pkl saved')
	
	def fit(self, X, y):
		tic = {}
		
		self.X = X
		self.y = y
		self.split_data()
		self.cluster_num = len(self.X_clusters.keys())
		self.feature_num = len(X[1])
		self.mean = {}

		# make the data in the right format
		for key in self.X_clusters.keys():
			data = self.X_clusters[key]
			if self.window_size >= 2:
				train_data = data[:-self.window_size+1]
				for i in range(self.window_size-1):
					data_window = np.concatenate([train_data, data[i+1:data.shape[0]-self.window_size+i+2]], axis=1)
					train_data = data_window
				
			else:
				train_data = data
				
			# train using ADMM
			self.mean[key] = np.mean(train_data, axis=0)[(self.window_size - 1) * self.feature_num:self.window_size * self.feature_num].reshape([1, self.feature_num])
			probSize = self.window_size * self.feature_num
			lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
			S = np.cov(np.transpose(train_data))
			solver = ADMMSolver(lamb, self.window_size, self.feature_num, 1, S)
			solver(1000, 1e-6, 1e-6, False)
			S_est = upperToFull(solver.x, 1e-6)
			X2 = S_est
			u, _ = np.linalg.eig(S_est)
			cov_out = np.linalg.inv(X2)
			cov_out = cov_out*100000/np.sum(cov_out)
			tic[key] = cov_out
			print('Cluster', key, 'calculated')
		self.tic = tic

	def split_data(self):
		X = self.X
		y = [str(label_) for label_ in (self.y)]
		self.X_clusters = {}
		for i, item in enumerate(y):
			if item in self.X_clusters.keys():
				self.X_clusters[item].append(X[i])
			else:
				self.X_clusters[item] = [X[i]]
		for i in self.X_clusters.keys():
			self.X_clusters[i] = np.array(self.X_clusters[i])
		
	def assign(self, window):
		lle = {}
		self.array_key = []
		for key in self.tic.keys():
			theta = self.tic[key]
			x = window - self.mean[key]
			XT = x.reshape([1, self.window_size * self.feature_num])
			X = x.reshape([self.feature_num * self.window_size, 1])
			logdet = np.log(np.linalg.det(theta))
			lle[key] = np.dot(XT, np.dot(theta, X)) - logdet / 2
		
		array = []
		for key in self.tic.keys():
			self.array_key.append(key)
			array.append(lle[key][0][0])
		return array
	
	def predict(self, data, beta=0.01, init_cluster=0):
		online_predict = []
		old = init_cluster
		lles = []
		for i in range(0, len(data) - self.window_size):
			temporal_consistency = np.array([beta] * self.cluster_num)
			lle = self.assign(data[i:self.window_size + i])
			_lle = copy.deepcopy(lle)
			lles.append(_lle)
			if old is not None:
				temporal_consistency[old] = 0
			lle += temporal_consistency * (np.sum(lle))
			_sorted = np.argsort(lle)
			now = _sorted[0]
			online_predict.append(int(float(self.array_key[now])))
			old = now
		self.predict_result = online_predict
		self.lles = lles
		# return online_predict, lles, top_2
	
	def optimize_tic(self):
		tic = self.tic
		for key in tic.keys():
			tic[key]


if __name__ == '__main__':
	data = Data()
	data.get_data('1.csv', '1.txt', feature_select=True)
	data.feature_preselect(k2=8)
	data2 = Data()
	data2.get_data('2.csv', '2.txt', feature_select=False)
	data2.select_feature(data.titles)
	X = data.data
	y = data.y
	X2 = data2.data
	y2 = data2.y
	X = np.concatenate((X, X2), axis=0)
	y = np.concatenate((y, y2), axis=0)
	
	oss = OnlineSmoothSegment(window_size=3, lambda_parameter=0.5)
	oss.fit(X, y)
	oss.save_MRFs()
	save_data(oss.tic, 'MRFs_p_'+str(oss.window_size)+'.pkl')
	test = Data(is_train_set=False)
	test.get_data('3.csv', '3.txt', feature_select=False)
	test.select_feature(data.titles)
	new_X = test.data
	new_y = test.y
	test2 = Data(is_train_set=False)
	test2.get_data('4.csv', '4.txt', feature_select=False)
	test2.select_feature(data.titles)
	new_X2 = test2.data
	new_y2 = test2.y
	new_X = np.concatenate((new_X, new_X2), axis=0)
	new_y = np.concatenate((new_y, new_y2), axis=0)
	oss.predict(new_X)
	
	plt.plot(oss.predict_result, label='预测值')
	plt.plot(new_y, label='真值')
	cnt = 1
	for i in range(len(oss.predict_result)):
		if oss.predict_result[i] == new_y[i]:
			cnt += 1
	acc = cnt/len(oss.predict_result)
	plt.xlabel('时间')
	plt.ylabel('类别')
	plt.title('Accuracy:'+str(acc))
	# plt.savefig('Results/results'+str(oss.window_size)+'.pdf', format='pdf')
	plt.show()
	