import matplotlib.pyplot as plt
import numpy as np


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



	