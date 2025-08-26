import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SR_py.model.pipeline import SissoRegressor
from SR_py.dsl.dimension import Dimension

def test_fit_accepts_numpy_arrays():
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    y = np.sin(X).ravel()
    model = SissoRegressor(K=1)
    model.fit(X, y)
    preds = model.predict(X[:2])
    assert preds.shape == (2,)


def test_missing_feature_dimensions_defaults_to_dimensionless():
    X = pd.DataFrame({'a': np.linspace(-1, 1, 5),
                      'b': np.linspace(0, 2, 5)})
    y = pd.Series(np.sin(X['a']))
    dims = {'a': Dimension([1, 0, 0, 0, 0, 0, 0])}
    model = SissoRegressor(K=1, dimensional_check=True)
    # Should not raise even though 'b' has no dimension provided
    model.fit(X, y, feature_dimensions=dims,
              target_dimension=Dimension([1, 0, 0, 0, 0, 0, 0]))
    assert model.feature_space_
