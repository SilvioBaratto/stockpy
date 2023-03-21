from ._lasso import LASSO as Lasso
from ._quantile import QUANTILE as Quantile
from ._sgd import SGD as SGD
from ._svr import SupportVector as SVR
from ._ridge import RIDGE as Ridge
from ._linear import LINEAR as Linear

__all__ = [
    "Lasso",
    "Quantile",
    "SGD",
    "SVR",
    "Ridge",
    "Linear"
]
