import numpy as np
X = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()
from SR_py.evolutionary.gp import GeneticProgramming
model = GeneticProgramming(population_size=50, n_generations=10)
model.fit(X, y)
print(model.explain())
from SR_py.evolutionary.ga_pso import GAPSORegressor
model = GAPSORegressor(generations=30)
model.fit(X, y)
print(model.explain())
from SR_py.sparse_regression.sisso import SISSORegressor
model = SISSORegressor(K=2)
model.fit(X, y)
print(model.explain())
from SR_py.sparse_regression.lasso_ridge_omp import LassoRegressor, RidgeRegressor, OMPRegressor
model = LassoRegressor()
model.fit(X, y)
print(model.explain())
from SR_py.sparse_regression.sindy import SINDyRegressor
model = SINDyRegressor()
model.fit(X, y)
print(model.explain())
from SR_py.probabilistic.bsr import BayesianSymbolicRegressor
model = BayesianSymbolicRegressor()
model.fit(X, y)
print(model.explain())
from SR_py.probabilistic.ppi import ProbabilisticProgramInduction
model = ProbabilisticProgramInduction()
model.fit(X, y)
print(model.explain())
from SR_py.neural_symbolic.deep_sr import DeepSymbolicRegression
model = DeepSymbolicRegression()
model.fit(X, y)
print(model.explain())
from SR_py.neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
model = NeuralSymbolicHybrid()
model.fit(X, y)
print(model.explain())
from SR_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
model = ReinforcementSymbolicRegression()
model.fit(X, y)
print(model.explain())
from SR_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
model = EvolutionaryGradientHybrid()
model.fit(X, y)
print(model.explain())
from SR_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
model = MultiObjectiveSymbolicRegression()
model.fit(X, y)
print(model.explain())
from SR_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
model = PhysicsInformedSymbolicRegression()
model.fit(X, y)
print(model.explain())


