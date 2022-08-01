import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from get_data import load_data


def connect_dots(x1, x2, y, w=2, color='black', alpha=0.2):
	xy = ((x1+x2)/2, y)
	r = (x2-x1)
	arc = Arc(xy, r, 2, theta1=0, theta2=180, color=color, linewidth=w)
	return arc


def mat2mrf(mat, window_size):
	plt.figure(figsize=(10, 4))
	plt.axis('off')
	fig = plt.gcf()
	ax = fig.gca()
	n = len(mat)
	n0 = int(n/window_size)
	gap = 1.2
	plt.xlim(-.5, n0-.5)
	plt.ylim(-.5, window_size+.6)
	# matrices = []
	color_table = ['royalblue', 'darkorange', 'darkgreen', 'darkred', 'darkviolet', 'darkcyan', 'darkmagenta', 'darkgray']
	for t in range(window_size):
		matrix = mat[t*n0:(t+1)*n0, :n0]
		max_val = matrix.max()
		threshold = 0
		nodes = [(i, t*gap) for i in range(len(matrix))]
		node_size = np.diagonal(matrix)
		node_size = node_size/(np.max(node_size)+1)
		plt.scatter(*zip(*nodes), s=node_size*300+50, c=color_table[t])
		if t == 0:
			for y in range(window_size):
				for i in range(len(matrix)):
					for j in range(len(matrix[i])):
						if i != j and np.abs(matrix[i][j]) > threshold:
							ax.add_patch(connect_dots(i, j, y*gap, 1+np.abs(matrix[i][j])/max_val*4, 'grey'))
		else:
			for y in range(window_size-t):
				for i in range(len(matrix)):
					for j in range(len(matrix[i])):
						if i != j and np.abs(matrix[i][j]) > threshold:
							plt.plot([i, j], [y*gap, (y+t)*gap], 'grey', alpha=0.5, linewidth=1+np.abs(matrix[i][j])/max_val*4)
	
	
# tic = load_data('Results/MRFs_3.pkl')[0]
tic = load_data('Results/MRFs_w=3lambda=0.01.pkl')[0]
# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
for i in range(len(tic)):
	mat2mrf(tic[i], 3)
	plt.tight_layout()
	plt.savefig('Results/vis_MRF_'+str(i)+'.pdf', format='pdf')
