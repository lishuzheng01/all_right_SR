"""Unified interface for AllRight-SR regressors.

This module exposes a simple factory function :func:`create_regressor`
that instantiates any available symbolic regressor with a consistent
API. All regressors share `fit` and `predict` methods compatible with the
scikit-learn style.
"""
from typing import Protocol, Dict, Type, Any

# Import all regressor classes
from .model.pipeline import SissoRegressor
from .evolutionary.island_gp import IslandGPRegressor
from .evolutionary.ga_pso import GAPSORegressor
from .sparse_regression.lasso_ridge_omp import LassoRegressor
from .sparse_regression.sindy import SINDyRegressor
from .probabilistic.bsr import BayesianSymbolicRegressor
from .probabilistic.ppi import ProbabilisticProgramInduction
from .neural_symbolic.rl_sr import ReinforcementSymbolicRegression
from .neural_symbolic.deep_sr import DeepSymbolicRegression
from .neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
from .hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
from .hybrid.physics_informed import PhysicsInformedSymbolicRegression
from .hybrid.multi_objective import MultiObjectiveSymbolicRegression


class SymbolicRegressor(Protocol):
    """Protocol describing common regressor methods."""

    def fit(self, X: Any, y: Any, *args, **kwargs) -> "SymbolicRegressor":
        ...

    def predict(self, X: Any, *args, **kwargs) -> Any:
        ...


_REGISTRY: Dict[str, Type[SymbolicRegressor]] = {
    "sisso": SissoRegressor,
    "gp": IslandGPRegressor,
    "ga_pso": GAPSORegressor,
    "lasso": LassoRegressor,
    "sindy": SINDyRegressor,
    "bsr": BayesianSymbolicRegressor,
    "ppi": ProbabilisticProgramInduction,
    "rl": ReinforcementSymbolicRegression,
    "dsr": DeepSymbolicRegression,
    "neural_hybrid": NeuralSymbolicHybrid,
    "evo_gradient": EvolutionaryGradientHybrid,
    "physics": PhysicsInformedSymbolicRegression,
    "multi_objective": MultiObjectiveSymbolicRegression,
}


def create_regressor(name: str, **kwargs) -> SymbolicRegressor:
    """Create a regressor by name.

    Parameters
    ----------
    name: str
        Identifier of the regressor. Available options are the keys of
        :data:`_REGISTRY`.
    **kwargs: dict
        Parameters forwarded to the regressor constructor.

    Returns
    -------
    SymbolicRegressor
        An instantiated regressor ready to ``fit`` and ``predict``.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown regressor '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](**kwargs)
