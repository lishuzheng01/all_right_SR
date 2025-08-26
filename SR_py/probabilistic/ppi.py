# -*- coding: utf-8 -*-
"""
概率程序归纳 (PPI) 符号回归实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import logging
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import time

from ..config import RANDOM_STATE
from ..utils.data_conversion import auto_convert_input, ensure_pandas_dataframe

logger = logging.getLogger(__name__)

class Rule:
    """表示概率程序中的规则"""
    def __init__(self, lhs, rhs, probability=1.0):
        self.lhs = lhs  # 左侧条件
        self.rhs = rhs  # 右侧表达式
        self.probability = probability  # 规则的概率
    
    def __str__(self):
        return f"{self.lhs} -> {self.rhs} [{self.probability:.2f}]"

class Expression:
    """表示符号表达式"""
    def __init__(self, op=None, args=None, value=None):
        self.op = op          # 操作符或函数名
        self.args = args or []  # 参数列表
        self.value = value    # 常数值或变量名
    
    def is_terminal(self):
        """是否为终端符号（变量或常数）"""
        return self.op is None
    
    def __str__(self):
        if self.is_terminal():
            return str(self.value)
        elif len(self.args) == 1:  # 一元操作符
            return f"{self.op}({str(self.args[0])})"
        elif len(self.args) == 2:  # 二元操作符
            return f"({str(self.args[0])} {self.op} {str(self.args[1])})"
        else:
            args_str = ", ".join(str(arg) for arg in self.args)
            return f"{self.op}({args_str})"
    
    def evaluate(self, X):
        """计算表达式的值"""
        if self.is_terminal():
            if isinstance(self.value, str):  # 变量
                return X[self.value].values
            else:  # 常数
                return np.full(len(X), self.value)
        
        # 评估参数
        args_values = [arg.evaluate(X) for arg in self.args]
        
        # 应用操作符
        if self.op == '+':
            return args_values[0] + args_values[1]
        elif self.op == '-':
            return args_values[0] - args_values[1]
        elif self.op == '*':
            return args_values[0] * args_values[1]
        elif self.op == '/':
            # 保护除法
            return np.divide(args_values[0], args_values[1], out=np.full_like(args_values[0], 1.0),
                           where=np.abs(args_values[1]) > 1e-10)
        elif self.op == 'sqrt':
            return np.sqrt(np.abs(args_values[0]))
        elif self.op == 'square':
            return args_values[0] ** 2
        elif self.op == 'log':
            return np.log(np.abs(args_values[0]) + 1e-10)
        elif self.op == 'exp':
            return np.exp(np.clip(args_values[0], -10, 10))
        elif self.op == 'sin':
            return np.sin(args_values[0])
        elif self.op == 'cos':
            return np.cos(args_values[0])
        else:
            return np.zeros(len(X))

class ProbabilisticProgramInduction(BaseEstimator, RegressorMixin):
    """
    基于概率程序归纳的符号回归实现
    
    PPI使用概率上下文无关文法来生成和评估符号表达式，
    并结合贝叶斯推断来学习最优表达式。
    
    参数:
    -----
    n_iterations : int, 默认=1000
        学习迭代次数
        
    population_size : int, 默认=100
        种群大小
        
    rule_complexity : int, 默认=3
        规则的最大复杂度
        
    max_expr_depth : int, 默认=5
        表达式的最大深度
        
    operators : List[str], 默认=None
        允许的操作符列表
        
    prior_temp : float, 默认=1.0
        先验温度参数
        
    mutation_prob : float, 默认=0.1
        变异概率
        
    crossover_prob : float, 默认=0.3
        交叉概率
        
    elite_fraction : float, 默认=0.1
        精英比例

    parsimony_coefficient : float, 默认=0.02
        简洁性系数

    random_state : int, 默认=42
        随机种子

    示例:
    -----
    >>> from SR_py.probabilistic.ppi import ProbabilisticProgramInduction
    >>> model = ProbabilisticProgramInduction()
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    def __init__(self,
                 n_iterations: int = 1000,
                 population_size: int = 100,
                 rule_complexity: int = 3,
                 max_expr_depth: int = 5,
                 operators: Optional[List[str]] = None,
                 prior_temp: float = 1.0,
                 mutation_prob: float = 0.1,
                 crossover_prob: float = 0.3,
                 elite_fraction: float = 0.1,
                 parsimony_coefficient: float = 0.02,
                 random_state: int = RANDOM_STATE):
        
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.rule_complexity = rule_complexity
        self.max_expr_depth = max_expr_depth
        self.operators = operators if operators else ['+', '-', '*', '/', 'sqrt', 'square', 'log', 'exp']
        self.prior_temp = prior_temp
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.elite_fraction = elite_fraction
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        
        # 将操作符分类
        self.unary_ops = ['sqrt', 'square', 'log', 'exp', 'sin', 'cos']
        self.binary_ops = ['+', '-', '*', '/']
        
        # 内部状态
        self._rules = []  # 概率规则集
        self._best_expr = None
        self._best_score = float('inf')
        self._feature_names = None
        self._fitted = False
    
    def _init_grammar(self):
        """初始化概率文法"""
        # 基本规则
        self._rules = []
        
        # 表达式 -> 终结符或非终结符
        self._rules.append(Rule('E', 'T', 0.6))  # 终结符
        self._rules.append(Rule('E', 'F', 0.4))  # 非终结符
        
        # 终结符规则
        # T -> 变量
        for feature in self._feature_names:
            self._rules.append(Rule('T', feature, 1.0 / len(self._feature_names)))
        
        # T -> 常数
        self._rules.append(Rule('T', 'C', 0.3))
        
        # 常数规则
        constants = [-2, -1, -0.5, 0, 0.5, 1, 2, 3.14]
        for const in constants:
            self._rules.append(Rule('C', const, 1.0 / len(constants)))
        
        # 函数规则
        # F -> 一元操作符(E)
        for op in self.unary_ops:
            if op in self.operators:
                self._rules.append(Rule('F', f"{op}(E)", 0.2 / len(self.unary_ops)))
        
        # F -> 二元操作符(E, E)
        for op in self.binary_ops:
            if op in self.operators:
                self._rules.append(Rule('F', f"({op}, E, E)", 0.3 / len(self.binary_ops)))
    
    def _generate_expr(self, symbol='E', depth=0):
        """生成符号表达式"""
        if depth >= self.max_expr_depth:
            # 深度限制，只允许终结符
            applicable_rules = [r for r in self._rules if r.lhs == 'T']
            if not applicable_rules:
                # 回退到常数
                return Expression(value=1.0)
            
            rule = random.choices(applicable_rules, 
                                [r.probability for r in applicable_rules])[0]
            
            if rule.rhs == 'C':
                # 生成常数
                const_rules = [r for r in self._rules if r.lhs == 'C']
                const_rule = random.choices(const_rules, 
                                         [r.probability for r in const_rules])[0]
                return Expression(value=const_rule.rhs)
            else:
                # 生成变量
                return Expression(value=rule.rhs)
        
        # 选择适用规则
        applicable_rules = [r for r in self._rules if r.lhs == symbol]
        if not applicable_rules:
            return None
        
        # 按概率选择规则
        rule = random.choices(applicable_rules, 
                            [r.probability for r in applicable_rules])[0]
        
        if symbol == 'T':
            if rule.rhs == 'C':
                # 生成常数
                const_rules = [r for r in self._rules if r.lhs == 'C']
                const_rule = random.choices(const_rules, 
                                         [r.probability for r in const_rules])[0]
                return Expression(value=const_rule.rhs)
            else:
                # 生成变量
                return Expression(value=rule.rhs)
        
        elif symbol == 'E':
            return self._generate_expr(rule.rhs, depth)
        
        elif symbol == 'F':
            # 解析函数规则
            if '(' in rule.rhs:
                parts = rule.rhs.split('(')
                op = parts[0]
                
                if op in self.unary_ops:
                    # 一元操作符
                    arg = self._generate_expr('E', depth + 1)
                    return Expression(op=op, args=[arg])
                elif ',' in parts[1]:
                    # 二元操作符
                    op = op.strip('(')
                    return Expression(op=op, args=[
                        self._generate_expr('E', depth + 1),
                        self._generate_expr('E', depth + 1)
                    ])
        
        # 默认情况
        return self._generate_expr('T', depth)
    
    def _evaluate_fitness(self, expr, X, y):
        """评估表达式适应度"""
        try:
            y_pred = expr.evaluate(X)
            mse = mean_squared_error(y, y_pred)
            
            # 计算表达式复杂度
            complexity = self._calculate_complexity(expr)
            
            # 添加复杂度惩罚
            penalized_score = mse + self.parsimony_coefficient * complexity
            
            return 1.0 / (1.0 + penalized_score)  # 转换为适应度，越大越好
        except Exception as e:
            logger.warning(f"评估表达式出错: {str(e)}")
            return 0.0
    
    def _calculate_complexity(self, expr):
        """计算表达式复杂度"""
        if expr.is_terminal():
            return 1
        else:
            return 1 + sum(self._calculate_complexity(arg) for arg in expr.args)
    
    def _mutate_expr(self, expr, depth=0):
        """变异表达式"""
        if expr.is_terminal() or random.random() < 0.5:
            # 替换表达式
            return self._generate_expr(depth=depth)
        
        # 递归变异子表达式
        new_args = []
        for arg in expr.args:
            if random.random() < self.mutation_prob:
                new_args.append(self._mutate_expr(arg, depth + 1))
            else:
                new_args.append(arg)
        
        return Expression(op=expr.op, args=new_args, value=expr.value)
    
    def _crossover_expr(self, expr1, expr2, depth=0):
        """交叉两个表达式"""
        if depth >= self.max_expr_depth or random.random() < 0.3:
            # 在深度限制处或随机情况下，直接交换表达式
            return expr2
        
        if expr1.is_terminal() or expr2.is_terminal():
            return expr2 if random.random() < 0.5 else expr1
        
        # 递归交叉
        new_args = []
        for i, arg in enumerate(expr1.args):
            if i < len(expr2.args) and random.random() < self.crossover_prob:
                new_args.append(self._crossover_expr(arg, expr2.args[i], depth + 1))
            else:
                new_args.append(arg)
        
        return Expression(op=expr1.op, args=new_args)
    
    def _update_rules(self, population, fitnesses):
        """基于种群更新概率规则"""
        # 计算每个规则的使用频率和成功概率
        rule_usage = {rule: 0 for rule in self._rules}
        rule_success = {rule: 0 for rule in self._rules}
        
        # 加权规则使用计数
        normalized_fitness = fitnesses / np.sum(fitnesses)
        
        for expr, fitness_weight in zip(population, normalized_fitness):
            used_rules = self._extract_rules_from_expr(expr)
            for rule in used_rules:
                if rule in rule_usage:
                    rule_usage[rule] += 1
                    rule_success[rule] += fitness_weight
        
        # 更新规则概率
        for lhs in ['E', 'T', 'C', 'F']:
            rules_for_lhs = [r for r in self._rules if r.lhs == lhs]
            total_usage = sum(rule_usage[r] for r in rules_for_lhs)
            
            if total_usage > 0:
                for rule in rules_for_lhs:
                    if rule_usage[rule] > 0:
                        # 贝叶斯更新
                        new_prob = (rule_success[rule] / total_usage) ** (1.0 / self.prior_temp)
                        rule.probability = new_prob
                
                # 归一化概率
                total_prob = sum(r.probability for r in rules_for_lhs)
                if total_prob > 0:
                    for rule in rules_for_lhs:
                        rule.probability /= total_prob
    
    def _extract_rules_from_expr(self, expr):
        """从表达式中提取使用的规则"""
        # 这个函数在实际实现中会更复杂
        # 这里只是一个简化版本
        used_rules = []
        
        if expr.is_terminal():
            if isinstance(expr.value, str):
                # 变量规则
                used_rules.extend([r for r in self._rules if r.lhs == 'T' and r.rhs == expr.value])
            else:
                # 常数规则
                used_rules.extend([r for r in self._rules if r.lhs == 'C' and r.rhs == expr.value])
        else:
            # 函数规则
            if len(expr.args) == 1:
                # 一元操作符
                used_rules.extend([r for r in self._rules if r.lhs == 'F' and r.rhs.startswith(f"{expr.op}(")])
            elif len(expr.args) == 2:
                # 二元操作符
                used_rules.extend([r for r in self._rules if r.lhs == 'F' and r.rhs.startswith(f"({expr.op},")])
            
            # 递归提取子表达式的规则
            for arg in expr.args:
                used_rules.extend(self._extract_rules_from_expr(arg))
        
        return used_rules
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        y: Union[np.ndarray, pd.Series, list],
    ):
        """拟合概率程序归纳模型"""
        logger.info("开始概率程序归纳训练...")

        X, y = auto_convert_input(X, y)

        # 初始化
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        self._feature_names = list(X.columns)
        self._init_grammar()
        
        # 初始化种群
        population = [self._generate_expr() for _ in range(self.population_size)]
        
        # 评估初始种群
        fitnesses = np.array([self._evaluate_fitness(expr, X, y) for expr in population])
        
        # 找到最佳个体
        best_idx = np.argmax(fitnesses)
        self._best_expr = population[best_idx]
        self._best_score = fitnesses[best_idx]
        
        # 迭代优化
        for iteration in range(self.n_iterations):
            start_time = time.time()
            
            # 选择（轮盘赌）
            selection_probs = fitnesses / np.sum(fitnesses)
            selected_indices = np.random.choice(
                len(population),
                size=self.population_size,
                p=selection_probs,
                replace=True
            )
            selected = [population[i] for i in selected_indices]
            
            # 精英保留
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            elites = [population[i] for i in elite_indices]
            
            # 创建新一代
            offspring = []
            
            # 添加精英
            offspring.extend(elites)
            
            # 生成后代
            while len(offspring) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                if random.random() < self.crossover_prob:
                    # 交叉
                    child = self._crossover_expr(parent1, parent2)
                else:
                    child = parent1
                
                if random.random() < self.mutation_prob:
                    # 变异
                    child = self._mutate_expr(child)
                
                offspring.append(child)
            
            # 限制种群大小
            offspring = offspring[:self.population_size]
            
            # 评估新种群
            fitnesses = np.array([self._evaluate_fitness(expr, X, y) for expr in offspring])
            
            # 更新最佳个体
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self._best_score:
                self._best_expr = offspring[best_idx]
                self._best_score = fitnesses[best_idx]
            
            # 更新种群
            population = offspring
            
            # 更新规则概率
            if iteration % 10 == 0:
                self._update_rules(population, fitnesses)
            
            # 输出进度
            if iteration % 50 == 0 or iteration == self.n_iterations - 1:
                end_time = time.time()
                logger.info(f"迭代 {iteration+1}/{self.n_iterations}, 最佳适应度: {self._best_score:.6f} (用时: {end_time-start_time:.2f}s)")
                logger.info(f"最佳表达式: {self._best_expr}")
        
        # 设置训练状态
        self._fitted = True
        
        # 计算最终模型性能
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"概率程序归纳训练完成，MSE: {mse:.6f}, R²: {r2:.6f}")
        logger.info(f"最终表达式: {self._best_expr}")
        self._train_X = X
        self._train_y = y

        return self
    
    def predict(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if not self._fitted or not self._best_expr:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        X = ensure_pandas_dataframe(X, feature_names=self._feature_names)

        return self._best_expr.evaluate(X)
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        -----
        Dict
            模型信息字典
        """
        if not self._fitted:
            return {"status": "模型尚未训练"}
        
        # 收集高概率规则
        high_prob_rules = {}
        for lhs in ['E', 'T', 'F']:
            rules_for_lhs = [r for r in self._rules if r.lhs == lhs]
            sorted_rules = sorted(rules_for_lhs, key=lambda r: r.probability, reverse=True)
            high_prob_rules[lhs] = [(r.rhs, r.probability) for r in sorted_rules[:5]]
        
        return {
            "expression": str(self._best_expr),
            "fitness": self._best_score,
            "complexity": self._calculate_complexity(self._best_expr),
            "high_prob_rules": high_prob_rules,
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
            "title": "PPI 回归分析报告",
            "configuration": {
                "population_size": self.population_size,
                "n_iterations": self.n_iterations
            },
            "results": {
                "final_model": {
                    "formula_latex": str(self._best_expr),
                    "formula_sympy": str(self._best_expr),
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
