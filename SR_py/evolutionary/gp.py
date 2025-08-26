# -*- coding: utf-8 -*-
"""
基于遗传编程的符号回归实现
"""

import logging
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from deap import base, creator, tools, gp, algorithms

from ..utils.parallel import get_parallel_backend
from ..metrics.regression import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from ..model.formatted_report import SissoReport as Report
from ..ops.base import Operator
from ..ops.algebra import Add, Sub, Mul, SafeDiv
from ..ops.power_root import Square, Pow, Sqrt, Cbrt
from ..ops.log_exp import SafeLog, Exp
from ..ops.abs_sign import Abs
from ..dsl.dimension import Dimension
from ..utils.logging import setup_logging
from ..config import RANDOM_STATE
from ..utils.data_conversion import auto_convert_input, ensure_numpy_array

logger = setup_logging()

def parallelize(func, iterable, n_jobs=-1):
    """
    Helper function to parallelize the evaluation of individuals using joblib.
    """
    from joblib import delayed
    parallel = get_parallel_backend(n_jobs=n_jobs)
    return parallel(delayed(func)(item) for item in iterable)


class GeneticProgramming:
    """
    A Genetic Programming model for symbolic regression.
    """
    def __init__(self,
                 population_size: int = 100,
                 n_generations: int = 20,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 tournament_size: int = 3,
                 min_depth: int = 2,
                 max_depth: int = 5,
                 operators: Optional[List[Operator]] = None,
                 n_jobs: int = -1,
                 random_state: int = RANDOM_STATE):
        
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.operators = operators
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._best_individual = None
        self._log = None
        self._feature_names = None
        self._pset = None
        self._toolbox = base.Toolbox()

        if self.n_jobs != 1:
            self._toolbox.register("map", parallelize, n_jobs=self.n_jobs)

    def fit(self, X, y, feature_names: Optional[List[str]] = None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        X_df, y_series = auto_convert_input(X, y, feature_names)
        feature_names = feature_names or X_df.columns.tolist()

        self._setup_primitives(feature_names)
        self._setup_toolbox()
        X_array = X_df.values
        y_array = y_series.values
        self._toolbox.register("evaluate", self._evaluate_individual, X=X_array, y=y_array)

        pop = self._toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self._toolbox, self.crossover_rate, self.mutation_rate, self.n_generations,
                                       stats=stats, halloffame=hof, verbose=True)

        self._best_individual = hof[0]
        self._log = log

        # 保存训练数据以便后续生成报告
        self._train_X = X_df
        self._train_y = y_series

        logger.info(f"Training finished. Best fitness: {self._best_individual.fitness.values[0]}")
        return self

    def _evaluate_individual(self, individual, X, y):
        try:
            func = self._toolbox.compile(expr=individual)
            # 转换X为行的形式，每一行是一个样本
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            # 对每个样本计算函数值
            y_pred = []
            for i in range(X_array.shape[0]):
                try:
                    # 为每个样本传递特征值作为函数参数
                    pred = func(*X_array[i, :])
                    y_pred.append(pred)
                except:
                    y_pred.append(np.inf)
            
            y_pred = np.array(y_pred)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return (np.inf,)
            mse = mean_squared_error(y, y_pred)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError, MemoryError):
            mse = np.inf
        return (mse,)

    def predict(self, X):
        if self._best_individual is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")
        func = self._toolbox.compile(expr=self._best_individual)

        X_array = ensure_numpy_array(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        y_pred = []
        for i in range(X_array.shape[0]):
            try:
                pred = func(*X_array[i, :])
                y_pred.append(pred)
            except:
                y_pred.append(0.0)  # 默认值

        return np.array(y_pred)

    def get_best_model_string(self):
        if self._best_individual is None:
            return "No model trained yet."
        return str(self._best_individual)

    def explain(self) -> Report:
        """生成包含评价指标的格式化报告"""
        if self._best_individual is None:
            return Report({"status": "Model not fitted."})

        metrics = {}
        try:
            y_pred = self.predict(self._train_X)
            mse = mean_squared_error(self._train_y, y_pred)
            metrics = {
                "train_mse": mse,
                "train_rmse": float(np.sqrt(mse)),
                "train_mae": mean_absolute_error(self._train_y, y_pred),
                "train_r2": r2_score(self._train_y, y_pred),
                "train_samples": len(self._train_y)
            }
        except Exception as e:
            metrics = {
                "train_mse": None,
                "train_rmse": None,
                "train_mae": None,
                "train_r2": None,
                "error": str(e)
            }

        report_data = {
            "configuration": {
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
            },
            "results": {
                "final_model": {
                    "formula_latex": str(self._best_individual),
                    "formula_sympy": str(self._best_individual),
                    "intercept": 0,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": 1
            }
        }

        return Report(report_data)

    def _setup_primitives(self, feature_names):
        self._feature_names = feature_names
        if self._feature_names is None:
            raise ValueError("Feature names must be provided.")
        self._pset = gp.PrimitiveSet("MAIN", len(self._feature_names))
        
        if self.operators is None:
            self.operators = [
                Add(), Sub(), Mul(), SafeDiv(),
                Square(), Pow(3), Sqrt(), Cbrt(),
                SafeLog(), Exp(), Abs()
            ]
            
        for op in self.operators:
            if isinstance(op, Operator):
                # Use the class name as the primitive name and __call__ method
                op_name = op.__class__.__name__.lower()
                self._pset.addPrimitive(op, op.arity, name=op_name)
            
        self._pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(self._feature_names)})

    def _setup_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self._toolbox.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=self.min_depth, max_=self.max_depth)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("compile", gp.compile, pset=self._pset)

        self._toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._pset)
