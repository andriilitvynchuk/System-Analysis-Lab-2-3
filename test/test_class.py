import sys
sys.path.append('..')
from backend import AdditiveModel

test = dict(dataset_size = 40,
			x_path = '../data/x.tsv',
			y_path = '../data/y.tsv',
			x_size = [2, 2, 2],
			y_size = 5,
			b_type = 'norm',
			polynom_type = 'hermit',
			polynom_degrees = [2, 2, 2],
			polynom_search = True,
			lambda_type = 'all')

am = AdditiveModel(**test)
am.find_additive_model()
print(am.get_coef_lambda())
print(am.get_coef_a())
print(am.get_coef_c())