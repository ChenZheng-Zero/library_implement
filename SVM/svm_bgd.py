import matplotlib.pyplot as plt
import numpy as np
import sys
import time


class svm_bgd:
	b = 0
	delta_cost_percent = 0.0
	k = 0
	learning_rate = 0.3 * 10 ** (-6)
	convergence_error_criterion = 0.25
	cost = []

	def __init__(self, C):
		self.C = C

	def fit(self, training_set, labels):
		self.training_set = training_set
		self.labels = labels
		self.d = len(training_set[0])
		self.W = np.zeros(self.d)
		start_time = time.time()
		self.cost.append(self._cost())
		while self.delta_cost_percent > self.convergence_error_criterion or self.k == 0:
			updated_W = self.W
			for j in range(self.d):
				updated_W[j] = self.W[j] - self.learning_rate * self._wj_gradient(j)
			self.W = updated_W
			self.b = self.b - self.learning_rate * self._b_gradient()
			self.k += 1
			new_cost = self._cost()
			self.delta_cost_percent = self._delta_cost_percent(new_cost)
			self.cost.append(new_cost)
		self.runtime = time.time() - start_time
		print("SVM BGD Convergency time: {}".format(self.runtime))
		# x = [i for i in range(self.k + 1)]
		# plt.plot(x, self.cost)
		# plt.show()	
		
	
	def _cost(self):
		cost = np.dot(self.W, self.W)/2.0
		for i in range(len(self.training_set)):
			cost += self.C * max(0.0, (1.0 - self.labels[i] * (np.dot(self.W, self.training_set[i]) + self.b)))
		return cost

	def _wj_gradient(self, j):
		wj = self.W[j]
		alpha = 0.0
		for i in range(len(self.training_set)):
			if self.labels[i] * (np.dot(self.W, self.training_set[i]) + self.b) < 1.0:
				alpha -= self.labels[i] * self.training_set[i][j]
		return wj + self.C * alpha

	def _b_gradient(self):
		beta = 0.0
		for i in range(len(self.training_set)):
			if self.labels[i] * (np.dot(self.W, self.training_set[i]) + self.b) < 1.0:
				beta -= self.labels[i]
		return self.C * beta

	def _delta_cost_percent(self, new_cost):
		delta_cost_percent = 100 * abs(new_cost - self.cost[-1]) / self.cost[-1]
		return delta_cost_percent

def read_training_set(training_set_filename):
	features = []
	with open(training_set_filename, 'r') as training_set_file:
		for line in training_set_file:
			feature = [int(s) for s in line.strip().split(',')]
			features.append(feature)
	return np.array(features)

def read_labels(labels_filename):
	labels = []
	with open(labels_filename, 'r') as labels_file:
		for line in labels_file:
			labels.append(int(line.strip()))
	return np.array(labels)	

if __name__ == '__main__':
	training_set = read_training_set(sys.argv[1])
	labels = read_labels(sys.argv[2])
	svm = svm_bgd(C=100)
	svm.fit(training_set, labels)


	