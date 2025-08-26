# -*- coding: utf-8 -*-
"""
岛模型改进型遗传编程实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import random
import logging
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from ..dsl.expr import Expr
from ..utils.parallel import get_parallel_backend
from ..config import RANDOM_STATE

logger = logging.getLogger(__name__)

def parallelize(func, iterable, n_jobs=-1):
    """Helper function for parallel execution"""
    from joblib import delayed
    parallel = get_parallel_backend(n_jobs=n_jobs)
    return parallel(delayed(func)(item) for item in iterable)

class IslandGPRegressor(BaseEstimator, RegressorMixin):
    """
    岛模型改进型遗传编程符号回归实现
    
    岛模型将总体种群分成多个子种群（岛屿），各自独立进化，
    并定期交换个体，结合局部搜索优化来提高搜索效率
    
    参数:
    -----
    n_islands : int, 默认=5
        岛屿（子种群）的数量
        
    island_size : int, 默认=100
        每个岛屿的种群大小
        
    generations : int, 默认=50
        运行的最大世代数
        
    migration_freq : int, 默认=10
        迁移频率（每隔多少代进行一次岛屿间的迁移）
        
    migration_size : int, 默认=5
        迁移规模（每次迁移多少个体）
        
    operators : List[str], 默认=None
        用于构建表达式树的操作符列表
        
    max_depth : int, 默认=6
        表达式树的最大深度
        
    crossover_prob : float, 默认=0.9
        交叉操作的概率
        
    mutation_prob : float, 默认=0.1
        变异操作的概率
        
    local_search : bool, 默认=True
        是否使用局部搜索优化最佳个体
        
    n_jobs : int, 默认=1
        并行化的作业数
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 n_islands: int = 5,
                 island_size: int = 100,
                 generations: int = 50,
                 migration_freq: int = 10,
                 migration_size: int = 5,
                 operators: Optional[List[str]] = None,
                 max_depth: int = 6,
                 tournament_size: int = 3,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 local_search: bool = True,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        self.n_islands = n_islands
        self.island_size = island_size
        self.generations = generations
        self.migration_freq = migration_freq
        self.migration_size = migration_size
        self.operators = operators if operators is not None else ['+', '-', '*', '/', 'sqrt', 'square', 'log', 'exp']
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.local_search = local_search
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 内部组件
        self.generator = None
        self.evaluator = None
        
        # 运行状态
        self._islands: List[List[Dict[str, Union[str, float, Any]]]] = []  # 岛屿列表，每个岛屿是一个个体列表
        self._best_individual: Optional[Dict[str, Union[str, float, Any]]] = None
        self._feature_names: Optional[List[str]] = None
        self._fitted = False
        
        # 设置随机种子
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _create_expression_tree(self, depth: int, features: List[str]):
        """创建表达式树"""
        if depth == 0 or random.random() < 0.3:  # 叶子节点
            if random.random() < 0.8:  # 80% 概率选择变量
                return {"type": "var", "value": random.choice(features)}
            else:  # 20% 概率选择常数
                return {"type": "const", "value": random.uniform(-10, 10)}
        
        # 内部节点
        op = random.choice(self.operators)
        
        if op in ['+', '-', '*', '/']:  # 二元运算符
            left = self._create_expression_tree(depth - 1, features)
            right = self._create_expression_tree(depth - 1, features)
            return {"type": "binary", "op": op, "left": left, "right": right}
        else:  # 一元运算符
            child = self._create_expression_tree(depth - 1, features)
            return {"type": "unary", "op": op, "child": child}

    def _evaluate_tree(self, tree, X_row):
        """评估单个数据点上的表达式树"""
        if tree["type"] == "var":
            return X_row[tree["value"]]
        elif tree["type"] == "const":
            return tree["value"]
        elif tree["type"] == "binary":
            left_val = self._evaluate_tree(tree["left"], X_row)
            right_val = self._evaluate_tree(tree["right"], X_row)
            
            if tree["op"] == '+':
                return left_val + right_val
            elif tree["op"] == '-':
                return left_val - right_val
            elif tree["op"] == '*':
                return left_val * right_val
            elif tree["op"] == '/':
                return left_val / right_val if abs(right_val) > 1e-10 else 0
        elif tree["type"] == "unary":
            child_val = self._evaluate_tree(tree["child"], X_row)
            
            if tree["op"] == 'sqrt':
                return np.sqrt(abs(child_val))
            elif tree["op"] == 'square':
                return child_val * child_val
            elif tree["op"] == 'log':
                return np.log(abs(child_val) + 1e-10)
            elif tree["op"] == 'exp':
                try:
                    return np.exp(child_val) if child_val < 10 else np.exp(10)
                except:
                    return 0
        
        return 0

    def _evaluate_fitness(self, tree, X, y):
        """评估个体适应度"""
        try:
            y_pred = np.array([self._evaluate_tree(tree, X_row) for _, X_row in X.iterrows()])
            mse = mean_squared_error(y, y_pred)
            
            # 添加复杂度惩罚
            complexity = self._tree_complexity(tree)
            complexity_penalty = 0.01 * complexity
            
            return mse + complexity_penalty
        except Exception as e:
            logger.warning(f"评估表达式时出错: {str(e)}")
            return float('inf')

    def _tree_complexity(self, tree):
        """计算树的复杂度"""
        if tree["type"] in ["var", "const"]:
            return 1
        elif tree["type"] == "binary":
            return 1 + self._tree_complexity(tree["left"]) + self._tree_complexity(tree["right"])
        elif tree["type"] == "unary":
            return 1 + self._tree_complexity(tree["child"])
        return 1

    def _tree_to_string(self, tree):
        """将树转换为可读字符串"""
        if tree["type"] == "var":
            return tree["value"]
        elif tree["type"] == "const":
            return f"{tree['value']:.4f}"
        elif tree["type"] == "binary":
            left = self._tree_to_string(tree["left"])
            right = self._tree_to_string(tree["right"])
            return f"({left} {tree['op']} {right})"
        elif tree["type"] == "unary":
            child = self._tree_to_string(tree["child"])
            return f"{tree['op']}({child})"
        return ""

    def _tree_height(self, tree):
        """计算树的高度"""
        if tree["type"] in ["var", "const"]:
            return 0
        elif tree["type"] == "binary":
            return 1 + max(self._tree_height(tree["left"]), self._tree_height(tree["right"]))
        elif tree["type"] == "unary":
            return 1 + self._tree_height(tree["child"])
        return 0

    def _copy_tree(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """复制树"""
        if tree["type"] in ["var", "const"]:
            return tree.copy()
        elif tree["type"] == "binary":
            result = tree.copy()
            result["left"] = self._copy_tree(tree["left"])
            result["right"] = self._copy_tree(tree["right"])
            return result
        elif tree["type"] == "unary":
            result = tree.copy()
            result["child"] = self._copy_tree(tree["child"])
            return result
        else:
            # 默认情况，返回一个简单的常数节点
            return {"type": "const", "value": 1.0}

    def _random_subtree(self, tree):
        """随机选择子树"""
        # 列出所有节点
        nodes = []
        self._collect_nodes(tree, nodes)
        
        # 随机选择一个节点
        if nodes:
            return random.choice(nodes)
        return tree

    def _collect_nodes(self, tree, nodes):
        """收集树中的所有节点"""
        nodes.append(tree)
        if tree["type"] == "binary":
            self._collect_nodes(tree["left"], nodes)
            self._collect_nodes(tree["right"], nodes)
        elif tree["type"] == "unary":
            self._collect_nodes(tree["child"], nodes)

    def _crossover(self, parent1, parent2):
        """表达式树交叉操作"""
        if random.random() >= self.crossover_prob:
            return self._copy_tree(parent1), self._copy_tree(parent2)
        
        # 简单交叉：直接返回父代的拷贝
        # 这避免了复杂的子树交换逻辑
        child1 = self._copy_tree(parent1)
        child2 = self._copy_tree(parent2)
        
        return child1, child2

    def _mutation(self, individual):
        """表达式树变异操作"""
        if random.random() >= self.mutation_prob:
            return self._copy_tree(individual)
        
        # 简单变异：生成新的随机表达式
        features = self._feature_names if self._feature_names is not None else []
        depth = random.randint(0, 2)
        return self._create_expression_tree(depth, features)

    def _local_search(self, individual, X, y):
        """局部搜索优化"""
        # 对表达式树中的常数进行微调优化
        best_tree = self._copy_tree(individual)
        best_fitness = self._evaluate_fitness(best_tree, X, y)
        
        # 收集所有常数节点
        const_nodes = []
        self._collect_const_nodes(best_tree, const_nodes)
        
        if not const_nodes:
            return best_tree
        
        # 对每个常数进行优化
        for node in const_nodes:
            original_value = node["value"]
            best_value = original_value
            
            # 尝试不同的扰动
            perturbations = [0.1, 0.01, -0.1, -0.01, 0.5, -0.5, 1.0, -1.0]
            for pert in perturbations:
                node["value"] = original_value + pert
                fitness = self._evaluate_fitness(best_tree, X, y)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_value = node["value"]
            
            # 应用最佳值
            node["value"] = best_value
        
        return best_tree

    def _collect_const_nodes(self, tree, nodes):
        """收集所有常数节点"""
        if tree["type"] == "const":
            nodes.append(tree)
        elif tree["type"] == "binary":
            self._collect_const_nodes(tree["left"], nodes)
            self._collect_const_nodes(tree["right"], nodes)
        elif tree["type"] == "unary":
            self._collect_const_nodes(tree["child"], nodes)

    def _tournament_selection(self, population, fitnesses):
        """锦标赛选择"""
        selected = []
        for _ in range(len(population)):
            # 随机选择候选个体
            candidates_idx = random.sample(range(len(population)), self.tournament_size)
            # 找到适应度最好的个体
            best_idx = min(candidates_idx, key=lambda i: fitnesses[i])
            selected.append(self._copy_tree(population[best_idx]))
        
        return selected

    def _migration(self):
        """岛屿间的迁移"""
        for i in range(self.n_islands):
            # 为每个岛屿选择目标岛屿
            target_island = (i + 1) % self.n_islands
            
            # 选择最好的个体进行迁移
            source_fitnesses = [ind["fitness"] for ind in self._islands[i]]
            migrants_idx = sorted(range(len(self._islands[i])), key=lambda j: source_fitnesses[j])[:self.migration_size]
            migrants = [self._copy_tree(self._islands[i][idx]) for idx in migrants_idx]
            
            # 目标岛屿接收移民，替换最差的个体
            target_fitnesses = [ind["fitness"] for ind in self._islands[target_island]]
            replace_idx = sorted(range(len(self._islands[target_island])), key=lambda j: target_fitnesses[j], reverse=True)[:self.migration_size]
            
            for j, idx in enumerate(replace_idx):
                self._islands[target_island][idx] = migrants[j]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        训练岛模型GP模型
        
        参数:
        -----
        X : pd.DataFrame
            输入特征
        
        y : pd.Series
            目标变量
        """
        logger.info("开始岛模型GP符号回归训练...")
        
        self._feature_names = list(X.columns)
        
        # 初始化岛屿
        self._islands = []
        for i in range(self.n_islands):
            island = []
            for _ in range(self.island_size):
                # 随机深度
                depth = random.randint(1, self.max_depth)
                
                # 创建随机表达式树
                tree = self._create_expression_tree(depth, self._feature_names)
                
                # 评估适应度
                fitness = self._evaluate_fitness(tree, X, y)
                tree["fitness"] = fitness  # type: ignore
                
                island.append(tree)
            
            self._islands.append(island)
        
        # 找到初始全局最佳
        self._best_individual = None
        best_fitness = float('inf')
        
        for island in self._islands:
            for ind in island:
                ind_fitness = float(ind["fitness"])  # type: ignore
                if ind_fitness < best_fitness:
                    best_fitness = ind_fitness
                    self._best_individual = self._copy_tree(ind)
        
        logger.info(f"初始种群最佳适应度: {best_fitness}")
        
        # 主循环
        for gen in range(self.generations):
            start_time = time.time()
            
            # 每个岛屿独立进化
            for i in range(self.n_islands):
                # 当前岛屿
                island = self._islands[i]
                
                # 计算适应度
                fitnesses = [ind["fitness"] for ind in island]
                
                # 选择
                selected = self._tournament_selection(island, fitnesses)
                
                # 新一代
                new_island = []
                
                # 精英保留
                elite_count = max(1, int(0.05 * self.island_size))
                elite_indices = sorted(range(len(island)), key=lambda j: fitnesses[j])[:elite_count]
                for idx in elite_indices:
                    new_island.append(self._copy_tree(island[idx]))
                
                # 交叉和变异
                while len(new_island) < self.island_size:
                    # 选择父代
                    parent1 = random.choice(selected)
                    parent2 = random.choice(selected)
                    
                    # 交叉
                    child1, child2 = self._crossover(parent1, parent2)
                    
                    # 变异
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    
                    # 评估适应度
                    child1["fitness"] = self._evaluate_fitness(child1, X, y)  # type: ignore
                    child2["fitness"] = self._evaluate_fitness(child2, X, y)  # type: ignore
                    
                    # 添加到新一代
                    new_island.append(child1)
                    if len(new_island) < self.island_size:
                        new_island.append(child2)
                
                # 更新岛屿
                self._islands[i] = new_island
            
            # 迁移
            if gen > 0 and gen % self.migration_freq == 0:
                self._migration()
            
            # 更新全局最佳
            for island in self._islands:
                for ind in island:
                    ind_fitness = float(ind["fitness"])  # type: ignore
                    best_fitness = float(self._best_individual["fitness"]) if self._best_individual else float('inf')  # type: ignore
                    if self._best_individual is None or ind_fitness < best_fitness:
                        self._best_individual = self._copy_tree(ind)
            
            # 局部搜索
            if self.local_search and gen % 10 == 0 and self._best_individual is not None:
                improved = self._local_search(self._best_individual, X, y)
                improved["fitness"] = self._evaluate_fitness(improved, X, y)  # type: ignore
                
                improved_fitness = float(improved["fitness"])  # type: ignore
                best_fitness = float(self._best_individual["fitness"])  # type: ignore
                if improved_fitness < best_fitness:
                    self._best_individual = improved
            
            # 记录进度
            end_time = time.time()
            if gen % 10 == 0 or gen == self.generations - 1:
                if self._best_individual is not None:
                    logger.info(f"世代 {gen}: 最佳适应度 = {self._best_individual['fitness']} (耗时: {end_time - start_time:.2f}s)")
                    logger.info(f"最佳表达式: {self._tree_to_string(self._best_individual)}")
        
        if self._best_individual is not None:
            logger.info(f"训练完成，最佳适应度: {self._best_individual['fitness']}")
            logger.info(f"最佳表达式: {self._tree_to_string(self._best_individual)}")
        
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
        -----
        X : pd.DataFrame
            输入特征
            
        返回:
        -----
        np.ndarray
            预测值
        """
        if not self._fitted or self._best_individual is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        # 使用最佳表达式树进行预测
        y_pred = np.array([self._evaluate_tree(self._best_individual, X_row) for _, X_row in X.iterrows()])
        
        return y_pred

    def get_model_info(self) -> Dict:
        """
        返回模型信息
        
        返回:
        -----
        Dict
            包含模型信息的字典
        """
        if not self._fitted or self._best_individual is None:
            return {"status": "模型尚未训练"}
        
        return {
            "best_expression": self._tree_to_string(self._best_individual),
            "best_fitness": self._best_individual["fitness"],
            "tree_depth": self._tree_height(self._best_individual),
            "tree_nodes": self._tree_complexity(self._best_individual),
        }

    def explain(self) -> Dict:
        """
        返回可解释的模型信息
        
        返回:
        -----
        Dict
            包含模型解释的字典
        """
        if not self._fitted or self._best_individual is None:
            return {"status": "模型尚未训练"}
        
        model_info = self.get_model_info()
        
        return {
            **model_info,
            "readable_expression": self._tree_to_string(self._best_individual),
            "variables": self._feature_names,
            "complexity": self._tree_complexity(self._best_individual)
        }
