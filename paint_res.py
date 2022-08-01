import matplotlib.pyplot as plt
import numpy as np
from get_data import load_data

predict_result, new_y = load_data('Results/buff.pkl')
for i, item in enumerate(predict_result):
	if item != new_y[i]:
		plt.barh(y=2.5, width=1, height=5, left=i, color='red', alpha=0.7)

plt.plot(predict_result, label='预测值', c='blue')
plt.plot(new_y, label='真值', c='black')
plt.legend()
cnt = 1
for i in range(len(predict_result)):
	if predict_result[i] == new_y[i]:
		cnt += 1
acc = cnt / len(predict_result)
plt.xlabel('时间')
plt.ylabel('类别')
plt.title('Accuracy:' + str(acc))
plt.savefig('Results/results.pdf', format='pdf')
plt.show()