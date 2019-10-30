import numpy as np
from scipy.special import eval_chebyt, eval_hermite, eval_legendre, eval_laguerre
from scipy.optimize import fmin_cg
from copy import deepcopy
from sklearn.linear_model import Ridge as LinearRegression
from tqdm import tqdm


def get_coef(X, y):
	reg = LinearRegression(fit_intercept=False, alpha=0.0001).fit(X, y)
	coef = reg.coef_
	return coef

# def get_coef(A, b):
# 	def f(x):
# 		return np.sum((b - np.dot(A, x)) ** 2)
# 	coef = fmin_cg(f, np.ones((A.shape[1])), maxiter=2000, disp=0)
# 	return coef

class AdditiveModel:

	def __init__(self, dataset_size, x_path, y_path, x_size, y_size, b_type, polynom_type, polynom_degrees,
		polynom_search, lambda_type):
		self.dataset_size = dataset_size
		self.x = np.loadtxt(x_path)
		self.y = np.loadtxt(y_path)
		self.x_size = x_size
		self.y_size = y_size
		self.y = self.y[:, :self.y_size]

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

		self.cache_y = deepcopy(self.y)
		for index in range(self.y.shape[1]):
			key = 'y' + str(index)
			self.cache_min_max[key] = [self.y[:, index].min(), self.y[:, index].max()]
			self.y[:, index] = (self.y[:, index] - self.cache_min_max[key][0]) / (self.cache_min_max[key][1] - self.cache_min_max[key][0])

	def set_b(self):
		if self.b_type == 'norm':
			self.b = deepcopy(self.y)
		elif self.b_type == 'min_max':
			self.b = np.zeros((self.y.shape[0], 1))
			for i in range(self.y.shape[0]):
				self.b[i, :] = (np.max(self.y[i, :]) + np.min(self.y[i, :])) / 2


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
		score = np.abs((b_mean - np.dot(X, get_coef(X, b_mean)))).mean()
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

	def find_coef_lambda_all(self):
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
		self.coef_lambda = get_coef(X, b_mean)
		self.X_coef_lambda = X

	def find_coef_lambda_separately(self):
		b_mean = np.mean(self.b, axis=1)
		coef_lambda = []
		X_coef_lambda = []
		pointer_x = 0
		# print('Error for L')
		for i in range(3):
			tmp_X = np.zeros((self.x.shape[0], (self.polynom_degrees[i] + 1) * self.x_size[i]))
			pointer_X = 0
			for j in range(self.x_size[i]):
				for d in range(self.polynom_degrees[i] + 1):
					tmp_X[:, pointer_X] = self.polynom_function(d, self.x[:, pointer_x])
					pointer_X += 1
				pointer_x += 1
			tmp_coef_lambda = get_coef(tmp_X, b_mean)
			# print(np.mean(np.abs(b_mean - np.dot(tmp_X, tmp_coef_lambda))))
			coef_lambda.append(tmp_coef_lambda)
			X_coef_lambda.append(tmp_X)

		self.X_coef_lambda = np.concatenate(X_coef_lambda, axis = 1)
		self.coef_lambda = np.concatenate(coef_lambda, axis = 0)


	def find_coef_a(self):
		coef_a = dict()
		X_coef_a = dict()
		# print('Errors for A')
		for index in range(self.y_size):
			tmp_y = self.y[:, index]
			coef_a[index] = []
			X_coef_a[index] = []
			pointer = 0
			for i in range(3):
				tmp_X = np.zeros((self.X_coef_lambda.shape[0], self.x_size[i]))
				for j in range(self.x_size[i]):
					polynom_subset = self.X_coef_lambda[:, pointer:pointer + self.polynom_degrees[i] + 1]
					lambda_coef_subset = self.coef_lambda[pointer:pointer + self.polynom_degrees[i] + 1]
					tmp_X[:, j] = np.dot(polynom_subset, lambda_coef_subset)
					pointer += self.polynom_degrees[i] + 1
				tmp_coef_a = get_coef(tmp_X, tmp_y)

				# print(np.mean(np.abs(tmp_y - np.dot(tmp_X, tmp_coef_a))))
				X_coef_a[index].append(tmp_X)
				coef_a[index].append(tmp_coef_a)

		self.coef_a = coef_a
		self.X_coef_a = X_coef_a
	
	def find_coef_c(self):
		coef_c = dict()
		X_coef_c = dict()
		# print('Errors for C')
		for index in range(self.y_size):
			tmp_y = self.y[:, index]
			tmp_X = np.zeros((self.X_coef_lambda.shape[0], 3))
			for i in range(3):
				tmp_X[:, i] = np.dot(self.X_coef_a[index][i], self.coef_a[index][i])
			tmp_coef_c = get_coef(tmp_X, tmp_y)
			X_coef_c[index] = tmp_X
			coef_c[index] = tmp_coef_c
			# print(np.mean(np.abs(tmp_y - np.dot(tmp_X, tmp_coef_c))))

		self.X_coef_c = X_coef_c
		self.coef_c = coef_c


	def find_additive_model(self):
		self.norm()
		self.set_b()

		if self.polynom_search:
			self.find_polynom_degrees()

		if self.lambda_type == 'all':
			self.find_coef_lambda_all()
		elif self.lambda_type == 'separately':
			self.find_coef_lambda_separately()

		self.find_coef_a()
		self.find_coef_c()

	def get_coef_lambda(self):
		string = ''
		pointer = 0
		for i in range(3):
			for j in range(self.x_size[i]):
				for d in range(self.polynom_degrees[i]):
					coef = self.coef_lambda[pointer]
					string += f'l_{i + 1}{j + 1}{d}={coef:.4f}  '
					pointer += 1
				string += '\n'
		return string

	def get_coef_a(self):
		string = ''
		for index in range(5):
			string += f'Y{index + 1} \n'
			for i, coef in enumerate(self.coef_a[index]):
				for j in range(len(coef)):
					string += f'a_{i + 1}{j + 1}={coef[j]:.4f} '
				string += '\n'
		return string

	def get_coef_c(self):
		string = ''
		for index in range(5):
			string += f'Y{index + 1} \n'
			for j in range(len(self.coef_c[index])):
				string += f'c_{j + 1}={self.coef_c[index][j]:.4f} '
			string += '\n'
		return string