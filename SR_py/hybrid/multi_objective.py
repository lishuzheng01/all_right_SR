# -*- coding: utf-8 -*-
"""
多目标符号回归方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class MultiObjectiveSymbolicRegression(BaseEstimator, RegressorMixin):
    """Multi-objective symbolic regression.

    Evolves expressions while optimising several criteria such as accuracy,
    complexity and interpretability. A Pareto front is maintained to expose
    trade-offs between objectives.

    Parameters
    ----------
    objectives : list[str], default=['accuracy', 'complexity', 'interpretability']
        Objectives to optimise. Supported values include ``'accuracy'``,
        ``'complexity'`` and ``'interpretability'``.
    population_size : int, default=50
        Number of individuals in the population.
    n_generations : int, default=20
        Evolutionary generations to run.
    pareto_front_size : int, default=10
        Number of non-dominated solutions retained after evolution.
    crossover_rate : float, default=0.8
        Probability of exchanging subtrees during crossover.
    mutation_rate : float, default=0.2
        Probability of mutating an individual.
    max_depth : int, default=6
        Maximum depth for generated expression trees.
    tournament_size : int, default=3
        Size of tournament used for parent selection.
    weights : list[float] or None, default=None
        Optional weighting of objectives. If ``None`` the NSGA-II algorithm is
        used to maintain diversity.
    random_state : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    >>> from SR_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
    >>> model = MultiObjectiveSymbolicRegression()
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    
    def __init__(self,
                 objectives=['accuracy', 'complexity', 'interpretability'],
                 population_size=50,
                 n_generations=20,
                 pareto_front_size=10,
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 max_depth=6,
                 tournament_size=3,
                 weights=None,  # 目标权重，如果为None则使用NSGA-II
                 random_state=42):
        
        self.objectives = objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.pareto_front_size = pareto_front_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.weights = weights
        self.random_state = random_state
        
        self.pareto_front_ = []
        self.best_compromise_solution_ = None
        self.evolution_history_ = []
        
    def fit(self, X, y, feature_names=None):
        """
        训练多目标符号回归模型
        """
        logger.info("开始多目标符号回归训练...")
        
        np.random.seed(self.random_state)
        
        self.feature_names_ = feature_names or [f'x{i}' for i in range(X.shape[1])]
        self.n_features_ = X.shape[1]
        
        # 初始化种群
        population = self._initialize_population()
        
        # 进化循环
        for generation in range(self.n_generations):
            # 评估种群
            objective_values = self._evaluate_population(population, X, y)
            
            # 计算帕累托前沿
            pareto_front_indices = self._calculate_pareto_front(objective_values)
            self.pareto_front_ = [population[i] for i in pareto_front_indices]
            
            # 记录历史
            self.evolution_history_.append({
                'generation': generation,
                'pareto_front_size': len(pareto_front_indices),
                'best_accuracy': min(obj[0] for obj in objective_values),
                'avg_complexity': np.mean([obj[1] for obj in objective_values]),
                'avg_interpretability': np.mean([obj[2] for obj in objective_values])
            })
            
            # 选择下一代
            population = self._nsga2_selection(population, objective_values)
            
            if generation % 5 == 0:
                logger.debug(f"Generation {generation}, Pareto front size: {len(pareto_front_indices)}")
        
        # 选择最佳妥协解
        self._select_best_compromise_solution(X, y)
        
        logger.info(f"训练完成，帕累托前沿大小: {len(self.pareto_front_)}")
        self._train_X = X
        self._train_y = y
        return self
    
    def predict(self, X):
        """预测"""
        if self.best_compromise_solution_ is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        return self._evaluate_expression(self.best_compromise_solution_, X)
    
    def _initialize_population(self):
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            expression = self._generate_random_expression()
            population.append(expression)
        
        return population
    
    def _generate_random_expression(self):
        """生成随机表达式"""
        templates = [
            "{var1}",
            "({var1} + {var2})",
            "({var1} * {var2})",
            "({var1} - {var2})",
            "({var1} / {var2})",
            "sin({var1})",
            "cos({var1})",
            "exp({var1} / 5)",
            "log(abs({var1}) + 1)",
            "sqrt(abs({var1}))",
            "({var1} ** 2)",
            "({var1} + {var2}) * {var3}",
            "sin({var1}) + cos({var2})",
            "({var1} * {var2}) + {var3}",
            "exp({var1}) / (1 + exp({var1}))"  # sigmoid-like
        ]
        
        template = np.random.choice(templates)
        
        # 随机选择变量
        selected_vars = list(np.random.choice(
            self.feature_names_,
            size=min(3, len(self.feature_names_)),
            replace=True
        ))

        # 一些模板需要 var2/var3，当特征数量不足时重复使用已有变量
        while len(selected_vars) < 3:
            selected_vars.append(selected_vars[0])

        var_mapping = {f'var{i+1}': var for i, var in enumerate(selected_vars)}

        return template.format(**var_mapping)
    
    def _evaluate_population(self, population, X, y):
        """评估种群的多目标函数值"""
        objective_values = []
        
        for expression in population:
            objectives = []
            
            for objective in self.objectives:
                if objective == 'accuracy':
                    obj_value = self._evaluate_accuracy(expression, X, y)
                elif objective == 'complexity':
                    obj_value = self._evaluate_complexity(expression)
                elif objective == 'interpretability':
                    obj_value = self._evaluate_interpretability(expression)
                else:
                    obj_value = 0.0
                
                objectives.append(obj_value)
            
            objective_values.append(objectives)
        
        return objective_values
    
    def _evaluate_accuracy(self, expression, X, y):
        """评估准确性（返回MSE，越小越好）"""
        try:
            y_pred = self._evaluate_expression(expression, X)
            return mean_squared_error(y, y_pred)
        except:
            return 1e6  # 惩罚无效表达式
    
    def _evaluate_complexity(self, expression):
        """评估复杂度（基于表达式长度和运算符数量）"""
        # 计算运算符数量
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'sqrt']
        complexity = 0
        
        for op in operators:
            complexity += expression.count(op)
        
        # 括号增加复杂度
        complexity += expression.count('(') * 0.5
        
        # 表达式长度
        complexity += len(expression) * 0.01
        
        return complexity
    
    def _evaluate_interpretability(self, expression):
        """评估可解释性（基于常见数学函数和简单结构）"""
        interpretability_score = 0
        
        # 线性项增加可解释性
        for var in self.feature_names_:
            if var in expression and ('*' + var not in expression and 
                                     var + '*' not in expression and
                                     'sin(' + var not in expression and
                                     'cos(' + var not in expression):
                interpretability_score += 1
        
        # 复杂函数降低可解释性
        complex_functions = ['exp', 'log', 'sin', 'cos', '**']
        for func in complex_functions:
            interpretability_score += expression.count(func) * 0.5
        
        # 嵌套函数降低可解释性
        nesting_level = 0
        max_nesting = 0
        for char in expression:
            if char == '(':
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif char == ')':
                nesting_level -= 1
        
        interpretability_score += max_nesting * 0.3
        
        return interpretability_score
    
    def _calculate_pareto_front(self, objective_values):
        """计算帕累托前沿"""
        pareto_front = []
        
        for i, obj_i in enumerate(objective_values):
            is_dominated = False
            
            for j, obj_j in enumerate(objective_values):
                if i != j and self._dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(i)
        
        return pareto_front
    
    def _dominates(self, obj1, obj2):
        """检查obj1是否支配obj2（所有目标都不差，且至少一个更好）"""
        better_in_at_least_one = False
        
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # obj1在目标i上更差
                return False
            elif obj1[i] < obj2[i]:  # obj1在目标i上更好
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _nsga2_selection(self, population, objective_values):
        """NSGA-II选择操作"""
        # 计算非支配排序
        fronts = self._non_dominated_sorting(objective_values)
        
        # 计算拥挤距离
        crowding_distances = self._calculate_crowding_distance(objective_values, fronts)
        
        # 选择下一代
        new_population = []
        front_index = 0
        
        while len(new_population) < self.population_size and front_index < len(fronts):
            front = fronts[front_index]
            
            if len(new_population) + len(front) <= self.population_size:
                # 整个前沿都可以加入
                new_population.extend([population[i] for i in front])
            else:
                # 按拥挤距离排序，选择部分个体
                remaining_slots = self.population_size - len(new_population)
                front_with_distances = [(i, crowding_distances[i]) for i in front]
                front_with_distances.sort(key=lambda x: x[1], reverse=True)
                
                for i, _ in front_with_distances[:remaining_slots]:
                    new_population.append(population[i])
            
            front_index += 1
        
        # 如果新种群不够，用交叉和变异生成
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                parent1 = np.random.choice(new_population)
                parent2 = np.random.choice(new_population)
                child = self._crossover(parent1, parent2)
            else:
                child = np.random.choice(new_population)
            
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _non_dominated_sorting(self, objective_values):
        """非支配排序"""
        n = len(objective_values)
        dominates = [[] for _ in range(n)]  # i支配的个体列表
        dominated_count = [0] * n  # 支配i的个体数量
        
        # 计算支配关系
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objective_values[i], objective_values[j]):
                        dominates[i].append(j)
                    elif self._dominates(objective_values[j], objective_values[i]):
                        dominated_count[i] += 1
        
        # 分层
        fronts = []
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            
            for i in current_front:
                for j in dominates[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    def _calculate_crowding_distance(self, objective_values, fronts):
        """计算拥挤距离"""
        n = len(objective_values)
        distances = [0.0] * n
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    distances[i] = float('inf')
                continue
            
            # 对每个目标计算拥挤距离
            for obj_idx in range(len(self.objectives)):
                # 按目标值排序
                front_sorted = sorted(front, key=lambda x: objective_values[x][obj_idx])
                
                # 边界个体设为无穷大
                distances[front_sorted[0]] = float('inf')
                distances[front_sorted[-1]] = float('inf')
                
                # 计算目标范围
                obj_range = (objective_values[front_sorted[-1]][obj_idx] - 
                           objective_values[front_sorted[0]][obj_idx])
                
                if obj_range == 0:
                    continue
                
                # 计算中间个体的拥挤距离
                for i in range(1, len(front_sorted) - 1):
                    distances[front_sorted[i]] += (
                        (objective_values[front_sorted[i+1]][obj_idx] - 
                         objective_values[front_sorted[i-1]][obj_idx]) / obj_range
                    )
        
        return distances
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        # 简单的字符串交叉
        if len(parent1) > 2 and len(parent2) > 2:
            point1 = np.random.randint(1, len(parent1))
            point2 = np.random.randint(1, len(parent2))
            
            child = parent1[:point1] + parent2[point2:]
            
            # 简单的语法修复
            if child.count('(') != child.count(')'):
                child = parent1  # 回退到父代
            
            return child
        else:
            return parent1
    
    def _mutate(self, individual):
        """变异操作"""
        # 随机替换一个变量
        for var in self.feature_names_:
            if var in individual:
                new_var = np.random.choice(self.feature_names_)
                individual = individual.replace(var, new_var, 1)
                break
        
        return individual
    
    def _select_best_compromise_solution(self, X, y):
        """从帕累托前沿选择最佳妥协解"""
        if not self.pareto_front_:
            self.best_compromise_solution_ = self._generate_random_expression()
            return
        
        if self.weights is not None:
            # 使用权重选择
            best_score = float('inf')
            best_solution = None
            
            for solution in self.pareto_front_:
                objectives = []
                objectives.append(self._evaluate_accuracy(solution, X, y))
                objectives.append(self._evaluate_complexity(solution))
                objectives.append(self._evaluate_interpretability(solution))
                
                # 加权求和
                weighted_score = sum(w * obj for w, obj in zip(self.weights, objectives))
                
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_solution = solution
            
            self.best_compromise_solution_ = best_solution
        else:
            # 选择准确性最好的解
            best_accuracy = float('inf')
            best_solution = None
            
            for solution in self.pareto_front_:
                accuracy = self._evaluate_accuracy(solution, X, y)
                if accuracy < best_accuracy:
                    best_accuracy = accuracy
                    best_solution = solution
            
            self.best_compromise_solution_ = best_solution
    
    def _evaluate_expression(self, expression, X):
        """评估表达式"""
        try:
            # 创建局部环境
            local_env = {}
            for i, name in enumerate(self.feature_names_):
                local_env[name] = X[:, i]
            
            # 添加数学函数
            local_env.update({
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-8),
                'sqrt': lambda x: np.sqrt(np.abs(x)),
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e
            })
            
            result = eval(expression, {"__builtins__": {}}, local_env)
            
            if np.isscalar(result):
                result = np.full(X.shape[0], result)
            
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        except:
            return np.zeros(X.shape[0])
    
    def get_pareto_front(self):
        """获取帕累托前沿"""
        return self.pareto_front_[:]
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'formula': self.best_compromise_solution_,
            'method': 'Multi-Objective Symbolic Regression',
            'objectives': self.objectives,
            'pareto_front_size': len(self.pareto_front_),
            'generations': len(self.evolution_history_),
            'compromise_solution': self.best_compromise_solution_
        }

    def _build_report(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.best_compromise_solution_ is None:
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
            "title": "Multi-Objective Symbolic Regression 分析报告",
            "configuration": {
                "objectives": self.objectives,
                "population_size": self.population_size,
                "n_generations": self.n_generations
            },
            "results": {
                "final_model": {
                    "formula_latex": self.best_compromise_solution_,
                    "formula_sympy": self.best_compromise_solution_,
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
