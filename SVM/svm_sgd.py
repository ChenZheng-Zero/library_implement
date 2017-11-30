import matplotlib.pyplot as plt
import numpy as np
import sys
import time


class svm_sgd:
	b = 0
	delta_cost = 0.0
	i = 1
	k = 0
	learning_rate = 0.1 * 10 ** (-3)
	convergence_error_criterion = 0.001
	cost = []

	def __init__(self, C):
		self.C = C

	def fit(self, training_set, labels):
		self._shuffle(training_set, labels)
		self.d = len(training_set[0])
		self.W = np.zeros(self.d)
		start_time = time.time()
		self.cost.append(self._cost())
		n = len(training_set)
		while self.delta_cost >= self.convergence_error_criterion or self.k == 0:
			updated_W = self.W
			for j in range(self.d):
				updated_W[j] = self.W[j] - self.learning_rate * self._wj_gradient(j)
			self.W = updated_W
			self.b = self.b - self.learning_rate * self._b_gradient()
			new_cost = self._cost()
			self.delta_cost = self._delta_cost(new_cost)
			self.cost.append(new_cost)			
			self.i = self.i % n + 1			
			self.k += 1
			# print(self.k, self.delta_cost)
		self.runtime = time.time() - start_time
		print("SVM SGD Convergency time: {}".format(self.runtime))
		# x = [i for i in range(self.k + 1)]
		# plt.plot(x, self.cost)		
		# plt.show()	
	
	def _shuffle(self, training_set, labels):
		indice = np.arange(len(training_set))
		np.random.shuffle(indice)
		self.training_set = np.zeros((len(training_set), len(training_set[0])))
		self.labels = np.zeros(len(labels))
		for i in range(len(indice)):
			self.training_set[i] = training_set[indice[i]]
			self.labels[i] = labels[indice[i]]
		return

		
	def _cost(self):
		cost = np.dot(self.W, self.W)/2.0
		for i in range(len(self.training_set)):
			cost += self.C * max(0.0, (1.0 - self.labels[i] * (np.dot(self.training_set[i], self.W) + self.b)))
		return cost

	def _wj_gradient(self, j):
		wj = self.W[j]
		gamma = 0.0
		if self.labels[self.i - 1] * (np.dot(self.W, self.training_set[self.i - 1]) + self.b) < 1.0:
			gamma = 0 - self.labels[self.i - 1] * self.training_set[self.i - 1][j]
		return wj + self.C * gamma

	def _b_gradient(self):
		delta = 0.0
		if self.labels[self.i - 1] * (np.dot(self.W, self.training_set[self.i - 1]) + self.b) < 1.0:
			delta = 0 - self.labels[self.i - 1]
		return self.C * delta

	def _delta_cost_percent(self, new_cost):
		delta_cost_percent = 100 * abs(new_cost - self.cost[-1]) / self.cost[-1]
		return delta_cost_percent

	def _delta_cost(self, new_cost):
		delta_cost_percent = self._delta_cost_percent(new_cost)
		return (self.delta_cost + delta_cost_percent)/2


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
	svm = svm_sgd(C=100)
	svm.fit(training_set, labels)


	