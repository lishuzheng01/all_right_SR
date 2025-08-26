import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SR_py.sparse_regression.lasso_ridge_omp import LassoRegressor

def test_explain_property_and_method_equivalence():
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    model = LassoRegressor()
    model.fit(X, y)
    report_call = model.explain()
    report_attr = model.explain
    assert str(report_call) == str(report_attr)
