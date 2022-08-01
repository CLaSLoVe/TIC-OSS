import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from get_data import load_data


tic = load_data('Results/MRFs_3.pkl')[0]
y = []
for key in tic.keys():
	y.append(np.sum(tic[key]))
plt.bar(range(len(tic.keys())), y)
plt.xlabel('类别')
plt.ylabel('求和')
plt.tight_layout()
plt.savefig('Results/MRFs_3.pdf', format='pdf')