# -*- coding: utf-8 -*-
"""
进化搜索与梯度优化混合方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class EvolutionaryGradientHybrid(BaseEstimator, RegressorMixin):
    """Hybrid evolutionary and gradient-based symbolic regression.

    The algorithm first performs a global evolutionary search to identify
    promising expression structures and then refines numeric constants using
    gradient descent.

    Parameters
    ----------
    evolution_phase_generations : int, default=20
        Number of generations in the evolutionary search phase.
    gradient_phase_iterations : int, default=100
        Iterations of gradient optimisation for each candidate.
    population_size : int, default=30
        Number of individuals maintained during evolution.
    learning_rate : float, default=0.01
        Step size used in the gradient refinement phase.
    momentum : float, default=0.9
        Momentum term for gradient updates.
    tolerance : float, default=1e-6
        Convergence threshold for gradient optimisation.
    max_expression_depth : int, default=6
        Maximum depth of generated expression trees.
    mutation_rate : float, default=0.1
        Probability of mutating an individual during evolution.
    crossover_rate : float, default=0.8
        Probability of performing crossover between individuals.
    elitism_rate : float, default=0.1
        Fraction of top individuals carried over unchanged to the next
        generation.
    random_state : int, default=42
        Seed for reproducibility.

    Examples
    --------
    >>> from SR_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
    >>> model = EvolutionaryGradientHybrid()
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    
    def __init__(self,
                 evolution_phase_generations=20,
                 gradient_phase_iterations=100,
                 population_size=30,
                 learning_rate=0.01,
                 momentum=0.9,
                 tolerance=1e-6,
                 max_expression_depth=6,
                 mutation_rate=0.1,
                 crossover_rate=0.8,
                 elitism_rate=0.1,
                 random_state=42):
        
        self.evolution_phase_generations = evolution_phase_generations
        self.gradient_phase_iterations = gradient_phase_iterations
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tolerance = tolerance
        self.max_expression_depth = max_expression_depth
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.random_state = random_state
        
        self.best_expression_ = None
        self.best_parameters_ = None
        self.best_score_ = float('inf')
        self.evolution_history_ = []
        self.gradient_history_ = []
        
    def fit(self, X, y, feature_names=None):
        """
        训练混合符号回归模型
        """
        logger.info("开始进化-梯度混合训练...")
        
        np.random.seed(self.random_state)
        
        self.feature_names_ = feature_names or [f'x{i}' for i in range(X.shape[1])]
        self.n_features_ = X.shape[1]
        
        # 第一阶段：进化搜索阶段
        logger.info("阶段1: 进化搜索...")
        best_candidates = self._evolutionary_phase(X, y)
        
        # 第二阶段：梯度优化阶段
        logger.info("阶段2: 梯度优化...")
        self._gradient_phase(X, y, best_candidates)
        
        logger.info(f"训练完成，最佳表达式: {self.best_expression_}")
        self._train_X = X
        self._train_y = y
        return self
    
    def predict(self, X):
        """预测"""
        if self.best_expression_ is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        return self._evaluate_expression_with_params(
            self.best_expression_, X, self.best_parameters_
        )
    
    def _evolutionary_phase(self, X, y):
        """进化搜索阶段"""
        # 初始化种群
        population = self._initialize_population()
        
        for generation in range(self.evolution_phase_generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                score = self._evaluate_individual(individual, X, y)
                fitness_scores.append(score)
                
                # 更新全局最佳
                if score < self.best_score_:
                    self.best_score_ = score
                    self.best_expression_ = individual['expression']
                    self.best_parameters_ = individual['parameters']
            
            # 记录历史
            self.evolution_history_.append({
                'generation': generation,
                'best_fitness': min(fitness_scores),
                'avg_fitness': np.mean(fitness_scores)
            })
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores)
            
            if generation % 5 == 0:
                logger.debug(f"Evolution generation {generation}, best fitness: {min(fitness_scores):.4f}")
        
        # 返回最优的几个候选
        elite_size = max(1, int(self.population_size * self.elitism_rate))
        sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        return [ind[0] for ind in sorted_pop[:elite_size]]
    
    def _gradient_phase(self, X, y, candidates):
        """梯度优化阶段"""
        best_candidate = None
        best_score = float('inf')
        
        for candidate in candidates:
            # 对每个候选进行梯度优化
            optimized_params, final_score = self._gradient_optimize(
                candidate, X, y
            )
            
            if final_score < best_score:
                best_score = final_score
                best_candidate = candidate.copy()
                best_candidate['parameters'] = optimized_params
        
        if best_candidate and best_score < self.best_score_:
            self.best_score_ = best_score
            self.best_expression_ = best_candidate['expression']
            self.best_parameters_ = best_candidate['parameters']
    
    def _gradient_optimize(self, individual, X, y):
        """对单个个体进行梯度优化"""
        params = individual['parameters'].copy()
        prev_params = params.copy()
        velocity = np.zeros_like(params)  # 动量项
        
        for iteration in range(self.gradient_phase_iterations):
            # 计算梯度（使用数值微分）
            gradients = self._compute_gradients(individual['expression'], X, y, params)
            
            # 更新参数（带动量的梯度下降）
            velocity = self.momentum * velocity - self.learning_rate * gradients
            params += velocity
            
            # 计算当前损失
            current_loss = self._evaluate_expression_loss(
                individual['expression'], X, y, params
            )
            
            # 记录历史
            self.gradient_history_.append({
                'iteration': iteration,
                'loss': current_loss,
                'params': params.copy()
            })
            
            # 检查收敛
            if np.linalg.norm(params - prev_params) < self.tolerance:
                break
                
            prev_params = params.copy()
        
        final_score = self._evaluate_expression_loss(
            individual['expression'], X, y, params
        )
        
        return params, final_score
    
    def _compute_gradients(self, expression, X, y, params, eps=1e-6):
        """使用数值微分计算梯度"""
        gradients = np.zeros_like(params)
        
        for i in range(len(params)):
            # 前向差分
            params_plus = params.copy()
            params_plus[i] += eps
            loss_plus = self._evaluate_expression_loss(expression, X, y, params_plus)
            
            params_minus = params.copy() 
            params_minus[i] -= eps
            loss_minus = self._evaluate_expression_loss(expression, X, y, params_minus)
            
            gradients[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return gradients
    
    def _initialize_population(self):
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            individual = {
                'expression': self._generate_random_expression(),
                'parameters': np.random.randn(3) * 0.5  # 初始参数
            }
            population.append(individual)
        
        return population
    
    def _generate_random_expression(self):
        """生成随机表达式"""
        templates = [
            "c0 * {var1} + c1 * {var2} + c2",
            "c0 * sin({var1}) + c1 * {var2}",
            "c0 * {var1} * {var2} + c1",
            "c0 * exp({var1} / 5) + c1 * {var2}",
            "c0 * {var1}**2 + c1 * {var2} + c2",
            "c0 * log(abs({var1}) + 1) + c1",
            "c0 * sqrt(abs({var1})) + c1 * {var2}",
            "c0 * cos({var1}) + c1 * sin({var2})"
        ]
        
        template = np.random.choice(templates)
        
        # 随机选择变量
        var1 = np.random.choice(self.feature_names_)
        var2 = np.random.choice(self.feature_names_)
        
        return template.format(var1=var1, var2=var2)
    
    def _evaluate_individual(self, individual, X, y):
        """评估个体适应度"""
        return self._evaluate_expression_loss(
            individual['expression'], X, y, individual['parameters']
        )
    
    def _evaluate_expression_loss(self, expression, X, y, params):
        """计算表达式的损失"""
        try:
            y_pred = self._evaluate_expression_with_params(expression, X, params)
            return mean_squared_error(y, y_pred)
        except:
            return 1e6  # 惩罚无效表达式
    
    def _evaluate_expression_with_params(self, expression, X, params):
        """使用参数评估表达式"""
        try:
            # 创建局部环境
            local_env = {}
            for i, name in enumerate(self.feature_names_):
                local_env[name] = X[:, i]
            
            # 添加参数
            for i, param in enumerate(params):
                local_env[f'c{i}'] = param
            
            # 添加数学函数
            local_env.update({
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-8),
                'sqrt': lambda x: np.sqrt(np.abs(x)),
                'abs': np.abs
            })
            
            result = eval(expression, {"__builtins__": {}}, local_env)
            
            if np.isscalar(result):
                result = np.full(X.shape[0], result)
            
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        except:
            return np.zeros(X.shape[0])
    
    def _evolve_population(self, population, fitness_scores):
        """进化种群"""
        # 选择精英
        elite_size = max(1, int(self.population_size * self.elitism_rate))
        sorted_indices = np.argsort(fitness_scores)
        new_population = [population[i] for i in sorted_indices[:elite_size]]
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                # 交叉
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
            else:
                # 直接选择
                child = self._tournament_selection(population, fitness_scores).copy()
            
            # 变异
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        tournament_indices = np.random.choice(
            len(population), size=tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        child = parent1.copy()
        
        # 参数交叉
        mask = np.random.random(len(parent1['parameters'])) < 0.5
        child['parameters'] = np.where(mask, parent1['parameters'], parent2['parameters'])
        
        return child
    
    def _mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        # 参数变异
        mutation_mask = np.random.random(len(individual['parameters'])) < 0.3
        noise = np.random.randn(len(individual['parameters'])) * 0.1
        mutated['parameters'] = individual['parameters'] + mutation_mask * noise
        
        return mutated
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'formula': self.best_expression_,
            'parameters': self.best_parameters_.tolist() if self.best_parameters_ is not None else None,
            'best_score': self.best_score_,
            'method': 'Evolutionary-Gradient Hybrid',
            'evolution_generations': len(self.evolution_history_),
            'gradient_iterations': len(self.gradient_history_)
        }

    def _build_report(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.best_expression_ is None:
            return SissoReport({"status": "Model not fitted."})

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

        report = {
            "title": "Evolutionary Gradient Hybrid 分析报告",
            "configuration": {
                "population_size": self.population_size,
                "evolution_generations": self.evolution_phase_generations,
                "gradient_iterations": self.gradient_phase_iterations
            },
            "results": {
                "final_model": {
                    "formula_latex": self.best_expression_,
                    "formula_sympy": self.best_expression_,
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

        return SissoReport(report)

    @property
    def explain(self):
        return self._build_report()
