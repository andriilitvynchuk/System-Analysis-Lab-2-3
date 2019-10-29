import numpy as np
from scipy.special import eval_chebyt, eval_hermite, eval_legendre, eval_laguerre
from copy import deepcopy
from sklearn.linear_model import Ridge as LinearRegression
from tqdm import tqdm

class AdditiveModel:

	def __init__(self, dataset_size, x_path, y_path, x_size, y_size, b_type, polynom_type, polynom_degrees,
		polynom_search, lambda_type):
		self.dataset_size = dataset_size
		self.x = np.loadtxt(x_path)
		self.y = np.loadtxt(y_path)
		self.x_size = x_size
		self.y_size = y_size

		if polynom_type == 'chebyshev':
			self.polynom_function = eval_chebyt
		elif polynom_type == 'hermit':
			self.polynom_function = eval_hermite
		elif polynom_type == 'legendre':
			self.polynom_function = eval_legendre
		elif polynom_type == 'laguerre':
			self.polynom_function = eval_laguerre

		self.b_type = b_type
		self.polynom_search = polynom_search
		if not self.polynom_search:
			self.polynom_degrees = polynom_degrees
		self.lambda_type = lambda_type

	def norm(self):
		self.cache_min_max = dict()
		for index in range(self.x.shape[1]):
			key = 'x' + str(index)
			self.cache_min_max[key] = [self.x[:, index].min(), self.x[:, index].max()]
			self.x[:, index] = (self.x[:, index] - self.cache_min_max[key][0]) / (self.cache_min_max[key][1] - self.cache_min_max[key][0])
		
		for index in range(self.y.shape[1]):
			key = 'y' + str(index)
			self.cache_min_max[key] = [self.y[:, index].min(), self.y[:, index].max()]
			self.y[:, index] = (self.y[:, index] - self.cache_min_max[key][0]) / (self.cache_min_max[key][1] - self.cache_min_max[key][0])

	def set_b(self):
		if self.b_type == 'norm':
			self.b = deepcopy(self.y)
		else:
			pass


	def evaluate_degrees(self, degrees):
		b_mean = np.mean(self.b, axis=1)
		new_shape_X = (self.x.shape[0], np.sum((np.array(degrees) + 1) * np.array(self.x_size)))
		X = np.zeros(new_shape_X)
		pointer_X = 0
		pointer_x = 0
		for i in range(3):
			for j in range(self.x_size[i]):
				for d in range(degrees[i] + 1):
					X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
					pointer_X += 1
				pointer_x += 1
		reg = LinearRegression(fit_intercept=False, alpha=0.0001).fit(X, b_mean)
		score = np.abs((b_mean - np.dot(X, reg.coef_))).mean()
		return score

	def find_polynom_degrees(self):
		degrees = [1, 1, 1]
		for index in tqdm(range(3)):
			best_score = np.inf
			best_degree = 1
			for degree in range(1, 6):
				degrees[index] = degree
				score = self.evaluate_degrees(degrees)
				if score < best_score:
					best_score = score
					best_degree = degree
			degrees[index] = best_degree
		print(degrees, best_score)
		self.polynom_degrees = degrees

	def find_lambda_all(self):
		b_mean = np.mean(self.b, axis=1)
		new_shape_X = (self.x.shape[0], np.sum((np.array(self.polynom_degrees) + 1) * np.array(self.x_size)))
		X = np.zeros(new_shape_X)
		pointer_X = 0
		pointer_x = 0
		for i in range(3):
			for j in range(self.x_size[i]):
				for d in range(self.polynom_degrees[i] + 1):
					X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
					pointer_X += 1
				pointer_x += 1
		regression = LinearRegression(fit_intercept=False, alpha=0.0001).fit(X, b_mean)
		self.X_lambda = X
		self.lambda_parameters = regression.coef_

	def find_additive_model(self):
		self.norm()
		self.set_b()

		if self.polynom_search:
			self.find_polynom_degrees()

		if self.lambda_type == 'all':
			self.find_lambda_all()
			print(self.lambda_parameters)
		elif self.lambda_type == 'separately':
			self.find_lambda_all()
			print(self.lambda_parameters)


