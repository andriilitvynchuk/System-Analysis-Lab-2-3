import sys

sys.path.append('..')
from backend import AdditiveModel

test = dict(dataset_size=40,
            x_path='../data/x.tsv',
            y_path='../data/y.tsv',
            x_size=[2, 2, 2],
            y_size=1,
            b_type='norm',
            polynom_type='chebyshev',
            polynom_degrees=[2, 2, 2],
            polynom_search=True,
            lambda_type='separately',
            output_file='./output.txt')

am = AdditiveModel(**test)
am.find_additive_model()
print(am.get_coef_lambda())
print(am.get_coef_a())
print(am.get_coef_c())
print(am.get_function_theta())
print(am.get_function_f_i())
print(am.get_final_approximation_f())
print(am.get_final_approximation_t())
print(am.get_final_approximation_polynoms())
print(am.get_final_approximation_polynoms_denorm())
am.write_in_file()
am.get_plot(norm=False)