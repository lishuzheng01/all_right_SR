import numpy as np
import pandas as pd

from sisso_py.evolutionary.gp import GeneticProgramming
from sisso_py.evolutionary.ga_pso import GAPSORegressor
from sisso_py.sparse_regression.sisso import SISSORegressor
from sisso_py.sparse_regression.lasso_ridge_omp import (
    LassoRegressor, RidgeRegressor, OMPRegressor,
)
from sisso_py.sparse_regression.sindy import SINDyRegressor
from sisso_py.probabilistic.bsr import BayesianSymbolicRegressor
from sisso_py.probabilistic.ppi import ProbabilisticProgramInduction
from sisso_py.neural_symbolic.deep_sr import DeepSymbolicRegression
from sisso_py.neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
from sisso_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
from sisso_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
from sisso_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
from sisso_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression


def _prepare_data():
    X = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
    X_df = pd.DataFrame(X, columns=["x"])
    y = pd.Series(np.sin(X).ravel())
    return X_df, y


def _assert_report(report):
    text = str(report)
    assert "y =" in text
    assert "MSE" in text
    assert "R²" in text


def test_all_models_explain():
    X_df, y = _prepare_data()

    gp = GeneticProgramming(population_size=20, n_generations=5)
    gp.fit(X_df.values, y.values, feature_names=["x"])
    _assert_report(gp.explain())

    ga_pso = GAPSORegressor(generations=5, population_size=20)
    ga_pso.fit(X_df, y)
    _assert_report(ga_pso.explain())

    sisso = SISSORegressor(K=1, sis_topk=10, so_max_terms=2)
    sisso.fit(X_df, y)
    _assert_report(sisso.explain())

    lasso = LassoRegressor()
    lasso.fit(X_df, y)
    _assert_report(lasso.explain())

    ridge = RidgeRegressor()
    ridge.fit(X_df, y)
    _assert_report(ridge.explain())

    omp = OMPRegressor(n_nonzero_coefs=2)
    omp.fit(X_df, y)
    _assert_report(omp.explain())

    sindy = SINDyRegressor()
    sindy.fit(X_df, y)
    _assert_report(sindy.explain())

    bsr = BayesianSymbolicRegressor(n_iter=50, n_chains=1, max_expr_depth=3)
    bsr.fit(X_df, y)
    _assert_report(bsr.explain())

    ppi = ProbabilisticProgramInduction(n_iterations=100, population_size=20, max_expr_depth=3)
    ppi.fit(X_df, y)
    _assert_report(ppi.explain())

    deep_sr = DeepSymbolicRegression(epochs=10)
    deep_sr.fit(X_df, y)
    _assert_report(deep_sr.explain())

    hybrid = NeuralSymbolicHybrid(epochs=10)
    hybrid.fit(X_df, y)
    _assert_report(hybrid.explain())

    rl = ReinforcementSymbolicRegression(max_episodes=20)
    rl.fit(X_df, y)
    _assert_report(rl.explain())

    evo_grad = EvolutionaryGradientHybrid(evolution_phase_generations=5, gradient_phase_iterations=10)
    evo_grad.fit(X_df, y)
    _assert_report(evo_grad.explain())

    multi = MultiObjectiveSymbolicRegression(n_generations=5, population_size=10)
    multi.fit(X_df, y)
    _assert_report(multi.explain())

    physics = PhysicsInformedSymbolicRegression(K=1)
    physics.fit(X_df, y)
    _assert_report(physics.explain())
