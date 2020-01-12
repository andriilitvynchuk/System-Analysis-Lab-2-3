import sys

from backend import AdditiveModel, MultiplyModel


sys.path.append("..")

test = dict(
    dataset_size=40,
    x_path="../data/x.tsv",
    y_path="../data/y.tsv",
    x_size=[2, 2, 2],
    y_size=1,
    b_type="norm",
    polynom_type="u",
    polynom_degrees=[2, 2, 2],
    polynom_search=True,
    lambda_type="separately",
    output_file="./output.txt",
)

am = MultiplyModel(**test)
# am = AdditiveModel(**test)
am.find_additive_model()
content = am.write_in_file()
print(content)
# am.get_plot(norm=True)
