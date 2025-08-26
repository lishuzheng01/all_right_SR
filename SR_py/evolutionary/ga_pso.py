# -*- coding: utf-8 -*-
"""
遗传算法与粒子群优化结合的符号回归实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional
import random
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ..utils.data_conversion import auto_convert_input, ensure_pandas_dataframe

from ..dsl.expr import Expr, Var
from ..gen.evaluator import FeatureEvaluator
from ..gen.generator import FeatureGenerator
from ..config import RANDOM_STATE

logger = logging.getLogger(__name__)

class Individual:
    """
    表示一个符号表达式个体
    """
    def __init__(self, expression: Expr, fitness: float = float('inf')):
        self.expression = expression
        self.fitness = fitness
        self.velocity = []  # PSO速度向量
        self.best_fitness = float('inf')
        self.best_expression = None

class GAPSORegressor(BaseEstimator, RegressorMixin):
    """
    遗传算法与粒子群优化结合的符号回归实现
    
    GA负责全局搜索和多样性维护，PSO负责局部搜索和快速收敛
    
    参数:
    -----
    population_size : int, 默认=200
        种群中的个体数量
        
    generations : int, 默认=50
        运行的最大世代数
        
    operators : List[str], 默认=None
        用于构建表达式的操作符列表，如果为None，则使用默认操作符
        
    max_depth : int, 默认=5
        表达式的最大深度
        
    crossover_prob : float, 默认=0.8
        交叉操作的概率
        
    mutation_prob : float, 默认=0.2
        变异操作的概率
        
    pso_c1 : float, 默认=1.5
        PSO个体认知参数
        
    pso_c2 : float, 默认=1.5
        PSO群体社会参数
        
    pso_w : float, 默认=0.7
        PSO惯性权重
        
    pso_update_freq : int, 默认=5
        PSO更新频率（多少代GA后进行一次PSO更新）
        
    n_jobs : int, 默认=1
        并行化的作业数

    random_state : int, 默认=42
        随机种子

    示例:
    -----
    >>> from SR_py.evolutionary.ga_pso import GAPSORegressor
    >>> model = GAPSORegressor(generations=30)
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    def __init__(self,
                 population_size: int = 200,
                 generations: int = 50,
                 operators: Optional[List[str]] = None,
                 max_depth: int = 5,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 pso_c1: float = 1.5,
                 pso_c2: float = 1.5,
                 pso_w: float = 0.7,
                 pso_update_freq: int = 5,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        self.population_size = population_size
        self.generations = generations
        self.operators = operators if operators is not None else ['+', '-', '*', '/', 'sqrt', 'square', 'log', 'exp']
        self.max_depth = max_depth
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_w = pso_w
        self.pso_update_freq = pso_update_freq
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 内部组件
        self.generator = None
        self.evaluator = None
        
        # 运行状态
        self._population = []
        self._global_best_individual = None
        self._feature_names = None
        self._fitted = False
        
        # 设置随机种子
        random.seed(self.random_state)
        np.random.seed(self.random_state)
    
    def _initialize_components(self, X: pd.DataFrame):
        """初始化符号回归组件"""
        self._feature_names = list(X.columns)
        
        # 初始化特征生成器
        self.generator = FeatureGenerator(
            operators=self.operators,
            max_complexity=self.max_depth * 3  # 大致对应深度
        )
        
        # 初始化评估器
        self.evaluator = FeatureEvaluator()
    
    def _initialize_population(self):
        """初始化种群"""
        # 创建初始变量
        initial_features = [Var(name) for name in self._feature_names]
        
        # 生成随机表达式
        population = []
        for _ in range(self.population_size):
            # 随机选择深度
            depth = random.randint(1, self.max_depth)
            
            # 生成随机表达式
            expr = self.generator.generate_random_expr(initial_features, depth)
            
            # 创建个体
            individual = Individual(expr)
            individual.velocity = [random.choice(self.operators) for _ in range(random.randint(1, 3))]
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, population: List[Individual], X: pd.DataFrame, y: pd.Series):
        """评估种群适应度"""
        # 提取表达式
        expressions = [ind.expression for ind in population]
        
        # 批量评估
        try:
            df_features, _ = self.evaluator.evaluate(expressions, X)
            
            # 计算适应度
            for i, individual in enumerate(population):
                expr_signature = individual.expression.get_signature()
                if expr_signature in df_features:
                    try:
                        y_pred = df_features[expr_signature]
                        mse = mean_squared_error(y, y_pred)
                        
                        # 添加复杂度惩罚
                        complexity = individual.expression.get_complexity()
                        complexity_penalty = 0.01 * complexity
                        
                        fitness = mse + complexity_penalty
                    except:
                        fitness = float('inf')
                else:
                    fitness = float('inf')
                
                individual.fitness = fitness
                
                # 更新个体最佳
                if fitness < individual.best_fitness:
                    individual.best_fitness = fitness
                    individual.best_expression = individual.expression
        
        except Exception as e:
            logger.warning(f"评估表达式时出错: {str(e)}")
    
    def _selection(self, population: List[Individual]) -> List[Individual]:
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            # 随机选择个体
            candidates = random.sample(population, 3)
            # 选择适应度最好的个体
            best = min(candidates, key=lambda ind: ind.fitness)
            selected.append(Individual(best.expression, best.fitness))
        
        return selected
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """表达式交叉操作"""
        if random.random() < self.crossover_prob:
            try:
                # 随机选择子表达式进行交换
                child1_expr = parent1.expression.swap_random_subexpr(parent2.expression)
                child2_expr = parent2.expression.swap_random_subexpr(parent1.expression)
                
                return Individual(child1_expr), Individual(child2_expr)
            except:
                # 如果交叉失败，返回父代
                return Individual(parent1.expression), Individual(parent2.expression)
        else:
            # 不进行交叉，直接返回父代
            return Individual(parent1.expression), Individual(parent2.expression)
    
    def _mutation(self, individual: Individual) -> Individual:
        """表达式变异操作"""
        if random.random() < self.mutation_prob:
            try:
                # 随机选择变异点
                mutated_expr = individual.expression.mutate()
                return Individual(mutated_expr)
            except:
                # 如果变异失败，返回原始个体
                return individual
        else:
            # 不进行变异
            return individual
    
    def _update_pso_velocity(self, population: List[Individual], global_best: Individual):
        """更新PSO速度"""
        for individual in population:
            # PSO速度更新: v = w*v + c1*r1*(pbest-p) + c2*r2*(gbest-p)
            r1, r2 = random.random(), random.random()
            
            # 这里的速度是一组操作符，我们使用一些启发式方法
            # 表示当前位置与pbest/gbest的"距离"
            
            # 使用随机运算符来表示速度
            new_velocity = []
            
            # 保持部分旧速度
            if individual.velocity and random.random() < self.pso_w:
                new_velocity.append(random.choice(individual.velocity))
            
            # 个体最佳的影响
            if individual.best_expression and random.random() < self.pso_c1 * r1:
                if individual.expression != individual.best_expression:
                    new_velocity.append(random.choice(self.operators))
            
            # 全局最佳的影响
            if global_best and random.random() < self.pso_c2 * r2:
                if individual.expression != global_best.expression:
                    new_velocity.append(random.choice(self.operators))
            
            # 更新速度，确保至少有一个操作符
            if not new_velocity:
                new_velocity.append(random.choice(self.operators))
            
            individual.velocity = new_velocity
    
    def _apply_pso_velocity(self, individual: Individual) -> Individual:
        """应用PSO速度更新个体位置"""
        expr = individual.expression
        
        # 应用速度中的每个操作符
        for op in individual.velocity:
            try:
                # 创建一个新的随机子表达式
                sub_expr = self.generator.generate_random_expr(
                    [Var(name) for name in self._feature_names], 
                    max_depth=2
                )
                
                # 根据操作符应用不同的变换
                if op in ['+', '-', '*', '/']:
                    # 二元操作，将当前表达式与新表达式结合
                    initial_features = [Var(name) for name in self._feature_names]
                    expr = self.generator.combine_expressions(expr, sub_expr, op)
                else:
                    # 一元操作，应用到表达式的随机子表达式
                    expr = expr.apply_op_to_random_subexpr(op)
            except:
                # 如果更新失败，保持不变
                pass
        
        return Individual(expr)
    
    def fit(self, X, y):
        """
        训练GA-PSO符号回归模型
        
        参数:
        -----
        X : pd.DataFrame
            输入特征
        
        y : pd.Series
            目标变量
        """
        logger.info("开始GA-PSO符号回归训练...")

        X_df, y_series = auto_convert_input(X, y)

        # 初始化组件
        self._initialize_components(X_df)

        # 初始化种群
        self._population = self._initialize_population()

        # 评估初始种群
        self._evaluate_population(self._population, X_df, y_series)
        
        # 找到初始全局最优
        self._global_best_individual = min(self._population, key=lambda ind: ind.fitness)
        
        logger.info(f"初始种群中最佳适应度: {self._global_best_individual.fitness}")
        
        # 主循环
        for gen in range(self.generations):
            # 选择
            selected = self._selection(self._population)
            
            # 新种群
            new_population = []
            
            # 交叉与变异
            i = 0
            while i < len(selected):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]
                    child1, child2 = self._crossover(parent1, parent2)
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    new_population.extend([child1, child2])
                else:
                    # 如果是奇数个体，最后一个直接变异
                    child = self._mutation(selected[i])
                    new_population.append(child)
                i += 2
            
            # PSO更新
            if gen % self.pso_update_freq == 0:
                # 更新速度
                self._update_pso_velocity(new_population, self._global_best_individual)
                
                # 应用速度更新位置
                pso_population = [self._apply_pso_velocity(ind) for ind in new_population]
                
                # 混合GA和PSO的结果
                new_population = new_population + pso_population
                
                # 选择前population_size个体
                new_population = sorted(new_population, key=lambda ind: ind.fitness)[:self.population_size]
            
            # 评估新种群
            self._evaluate_population(new_population, X_df, y_series)
            
            # 更新种群
            self._population = new_population
            
            # 更新全局最优
            best_in_gen = min(self._population, key=lambda ind: ind.fitness)
            if best_in_gen.fitness < self._global_best_individual.fitness:
                self._global_best_individual = Individual(best_in_gen.expression, best_in_gen.fitness)
            
            # 记录进度
            if gen % 10 == 0 or gen == self.generations - 1:
                logger.info(f"世代 {gen}: 最佳适应度 = {self._global_best_individual.fitness}")
        
        logger.info(f"训练完成，最佳适应度: {self._global_best_individual.fitness}")
        logger.info(f"最佳表达式: {self._global_best_individual.expression}")

        self._train_X = X_df
        self._train_y = y_series
        self._fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        使用训练好的模型进行预测

        参数:
        -----
        X : array-like
            输入特征

        返回:
        -----
        np.ndarray
            预测值
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        X_df = ensure_pandas_dataframe(X, feature_names=self._feature_names)

        # 评估最佳表达式
        df_result, _ = self.evaluator.evaluate([self._global_best_individual.expression], X_df)
        expr_signature = self._global_best_individual.expression.get_signature()

        return df_result[expr_signature].values
    
    def get_model_info(self) -> Dict:
        """
        返回模型信息
        
        返回:
        -----
        Dict
            包含模型信息的字典
        """
        if not self._fitted:
            return {"status": "模型尚未训练"}
        
        return {
            "best_expression": str(self._global_best_individual.expression),
            "best_fitness": self._global_best_individual.fitness,
            "expression_complexity": self._global_best_individual.expression.get_complexity(),
        }
    
    def _build_report(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if not self._fitted:
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
            "title": "GA-PSO 回归分析报告",
            "configuration": {
                "population_size": self.population_size,
                "generations": self.generations,
                "pso_update_freq": self.pso_update_freq
            },
            "results": {
                "final_model": {
                    "formula_latex": str(self._global_best_individual.expression),
                    "formula_sympy": str(self._global_best_individual.expression),
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
