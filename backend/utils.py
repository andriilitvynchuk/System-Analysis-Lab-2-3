from scipy.optimize import fmin_cg
from scipy.special import eval_chebyt, eval_hermite, eval_laguerre, eval_legendre
from sklearn.linear_model import Ridge as LinearRegression


def get_coef(X, y):
    reg = LinearRegression(fit_intercept=False, alpha=0.0001).fit(X, y)
    coef = reg.coef_
    return coef


def get_coef_cg(A, b):
    coef = fmin_cg(
        lambda x: np.sum((b - np.dot(A, x)) ** 2),
        np.ones((A.shape[1])),
        maxiter=2000,
        disp=0,
    )
    return coef


def eval_u(d, vector):
    return eval_chebyt(d, vector) + 5 * vector ** d


def eval_c(d, vector):
    return eval_hermite(d, vector) + 5 * vector ** d


def eval_s(d, vector):
    return eval_legendre(d, vector) + 5 * vector ** d


def eval_custom(d, vector):
    return eval_legendre(d, vector) + 5 * vector ** d - 5 * vector ** (1 + d)
