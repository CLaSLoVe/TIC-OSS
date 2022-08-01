import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = '/Users/clas/Downloads/Work/Code/CognitiveStatusResearch/Dat/jm2/FlightParameter/'
LABEL_ROOT = 'label/'

Ver = 'English'

if Ver == 'English':
	import vocabulary.vocabulary_list_en as vl
elif Ver == 'Chinese':
	import vocabulary.vocabulary_list_cn as vl


def save_data(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def load_data(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data


def tanh_normalize(data, par=None):
	max_for_each_column = np.max(data, axis=0)
	min_for_each_column = np.min(data, axis=0)
	if par:
		c = par[0]
		d = par[1]
	else:
		c = (max_for_each_column + min_for_each_column) / 2
		d = (max_for_each_column - min_for_each_column) / 2
	for i in range(data.shape[1]):
		if max_for_each_column[i] != min_for_each_column[i]:
			data[:, i] = 1 / 2 * np.tanh(2 * (data[:, i] - c[i]) / d[i]) + 1 / 2
		else:
			data[:, i] = [0] * data.shape[0]
	return data, (c, d)


class Data:
	def __init__(self, is_train_set=True, norm_parameter=None):
		self.is_train_set = is_train_set
		self.norm_parameter = norm_parameter
	
	def get_data(self, filename, labelname=None, feature_select=False):
		self.filename = filename
		# get X
		with open(ROOT + filename, 'r', encoding='gb2312') as f:
			self.titles = f.readline().strip().split(',')
		self.data = np.loadtxt(ROOT + filename, delimiter=',', skiprows=1, encoding='gb2312')
		
		if len(self.data[0]) < len(self.titles):
			print('Error: data length is less than titles length, self.titles is cut.')
			self.titles = self.titles[:len(self.data[0])]
		# scale X
		# scaler = MinMaxScaler()
		# self.data = scaler.fit_transform(self.data)
		if self.is_train_set:
			self.data, self.norm_parameter = tanh_normalize(self.data)
		else:
			self.data, self.norm_parameter = tanh_normalize(self.data, self.norm_parameter)
		print('Data obtained. size:', self.data.shape)
		if labelname:
			# get y
			self.get_label(labelname)
			feature_select = False
		# feature select
		self.del_feature(['时间ID', '剩油量'])
		if feature_select:
			self.del_constant()
			self.feature_preselect()
		print('Data preselected. size:', self.data.shape)
	
	def del_feature(self, name_list=None):
		# delete features
		print('Delete features:', name_list)
		index_list = [self.titles.index(name) for name in name_list]
		self.data = np.delete(self.data, index_list, axis=1)
		self.titles = [self.titles[i] for i in range(len(self.titles)) if i not in index_list]
	
	def select_feature(self, name_list=None):
		# select features
		print('Select features:', name_list)
		if name_list is None:
			name_list = self.titles
		index_list = [self.titles.index(name) for name in name_list]
		self.titles = [self.titles[i] for i in index_list]
		self.data = self.data[:, index_list]
		print('Data selected. size:', self.data.shape)
	
	def del_constant(self):
		# delete constant features
		print('Delete constant features')
		min_ = self.data.min(axis=0)
		max_ = self.data.max(axis=0)
		gaps = max_ - min_
		vary_index = np.where(gaps != 0)[0]
		self.data = self.data[:, vary_index]
		self.titles = [self.titles[i] for i in vary_index]
		print('Data deleted. size:', self.data.shape)
	
	def feature_preselect(self, k1=0.8, k2=8):
		# feature select
		print('VarianceThreshold...')
		try:
			selector = VarianceThreshold(threshold=k1)
			self.data = selector.fit_transform(self.data)
			self.selected_index = selector.get_support(True)
			self.titles = [self.titles[i] for i in self.selected_index]
			print('VarianceThreshold done.')
		except ValueError:
			print('VarianceThreshold error. Try SelectKBest.')
		print('SelectKBest...')
		selector = SelectKBest(chi2, k=k2)
		self.data = selector.fit_transform(self.data, self.y)
		self.selected_index = selector.get_support(True)
		self.titles = [self.titles[i] for i in self.selected_index]
		print('SelectKBest done.')
	
	def plot_data(self, norm=True):
		plt.figure(figsize=(8, 5))
		if self.labeled:
			plt.vlines(self.labels, ymin=-0.04, ymax=+1.04, colors='r', linestyles='--', label=vl.stages)
		if norm:
			min_ = self.data.min(axis=0)
			max_ = self.data.max(axis=0)
			data = (self.data - min_) / (max_ - min_)
		plt.plot(data, label=vl.titles(self.titles))
		plt.legend(loc='best')
		plt.legend(loc=1, bbox_to_anchor=(1, -.3), borderaxespad=0., ncol=4)
		plt.xlabel(vl.time)
		plt.ylabel(vl.fps)
		plt.tight_layout()
		plt.savefig('Results/pic/vis_fp/'+self.filename+'_fp.pdf', format='pdf')
		plt.show()
	
	def get_label(self, filename):
		# get y
		self.labeled = True
		self.labels = np.loadtxt(LABEL_ROOT + filename, delimiter=',')
		self.labels = list(self.labels) + [len(self.data)]
		ground_truth = []
		labels = [0] + self.labels
		for i in range(len(labels) - 1):
			ground_truth.extend([i] * int(labels[i + 1] - labels[i]))
		self.y = np.array(ground_truth)


if __name__ == '__main__':
	# ROOT = ''
	# LABEL_ROOT = 'label/'
	self = Data()
	self.get_data('1.csv', '1.txt', feature_select=False)
	self.feature_preselect(k2=15)
	# self.select_feature(['飞行模式', '方位', '海拔', 'EO方位角', 'EO俯仰角', 'EO焦距'])
	self.plot_data()
