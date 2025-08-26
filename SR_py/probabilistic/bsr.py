# -*- coding: utf-8 -*-
"""
贝叶斯符号回归实现 (BSR, MCMC)
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

from ..dsl.expr import Var, Expr
from ..config import RANDOM_STATE
from ..utils.data_conversion import auto_convert_input, ensure_pandas_dataframe

logger = logging.getLogger(__name__)

class Expression:
    """表达式类，用于MCMC采样"""
    def __init__(self, expr_type=None, value=None, left=None, right=None):
        self.type = expr_type  # 'const', 'var', 'unary_op', 'binary_op'
        self.value = value  # 常数值、变量名或操作符
        self.left = left    # 左子树
        self.right = right  # 右子树
        self.depth = self._calculate_depth()  # 树的深度
        self.size = self._calculate_size()    # 树的节点数
    
    def _calculate_depth(self):
        """计算树的深度"""
        if self.type in ['const', 'var']:
            return 0
        elif self.type == 'unary_op':
            return 1 + self.left.depth
        elif self.type == 'binary_op':
            return 1 + max(self.left.depth, self.right.depth)
        return 0
    
    def _calculate_size(self):
        """计算树的节点数"""
        if self.type in ['const', 'var']:
            return 1
        elif self.type == 'unary_op':
            return 1 + self.left.size
        elif self.type == 'binary_op':
            return 1 + self.left.size + self.right.size
        return 0
    
    def evaluate(self, X):
        """评估表达式在给定数据上的值"""
        if self.type == 'const':
            return np.full(len(X), self.value)
        elif self.type == 'var':
            return X[self.value].values
        elif self.type == 'unary_op':
            x = self.left.evaluate(X)
            if self.value == 'sqrt':
                return np.sqrt(np.abs(x))  # 避免负数
            elif self.value == 'square':
                return x ** 2
            elif self.value == 'log':
                return np.log(np.abs(x) + 1e-10)  # 避免对0取对数
            elif self.value == 'exp':
                return np.exp(np.clip(x, -10, 10))  # 避免溢出
            elif self.value == 'sin':
                return np.sin(x)
            elif self.value == 'cos':
                return np.cos(x)
            elif self.value == 'abs':
                return np.abs(x)
            elif self.value == 'neg':
                return -x
        elif self.type == 'binary_op':
            left_val = self.left.evaluate(X)
            right_val = self.right.evaluate(X)
            if self.value == '+':
                return left_val + right_val
            elif self.value == '-':
                return left_val - right_val
            elif self.value == '*':
                return left_val * right_val
            elif self.value == '/':
                return np.divide(left_val, right_val, out=np.ones_like(left_val), where=np.abs(right_val) > 1e-10)
        return np.zeros(len(X))
    
    def to_string(self):
        """转换为字符串表示"""
        if self.type == 'const':
            return f"{self.value:.4f}"
        elif self.type == 'var':
            return self.value
        elif self.type == 'unary_op':
            if self.value == 'square':
                return f"({self.left.to_string()})^2"
            else:
                return f"{self.value}({self.left.to_string()})"
        elif self.type == 'binary_op':
            return f"({self.left.to_string()} {self.value} {self.right.to_string()})"
        return ""
    
    def copy(self):
        """创建表达式的深拷贝"""
        if self.type in ['const', 'var']:
            return Expression(self.type, self.value)
        elif self.type == 'unary_op':
            return Expression(self.type, self.value, self.left.copy())
        elif self.type == 'binary_op':
            return Expression(self.type, self.value, self.left.copy(), self.right.copy())
        return None
    
    def get_all_nodes(self):
        """获取所有节点"""
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_all_nodes())
        if self.right:
            nodes.extend(self.right.get_all_nodes())
        return nodes
    
    def replace_node(self, old_node, new_node):
        """替换节点"""
        if self.left is old_node:
            self.left = new_node
            return True
        if self.right is old_node:
            self.right = new_node
            return True
        
        if self.left and self.left.replace_node(old_node, new_node):
            return True
        if self.right and self.right.replace_node(old_node, new_node):
            return True
        
        return False
    
    def get_complexity(self):
        """获取表达式复杂度"""
        # 基于大小和深度的加权和
        return self.size * 0.7 + self.depth * 0.3

class BayesianSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    基于贝叶斯推断的符号回归实现
    
    采用马尔可夫链蒙特卡洛 (MCMC) 方法进行参数和结构的贝叶斯推断
    
    参数:
    -----
    n_iter : int, 默认=10000
        MCMC迭代次数
        
    burn_in : int, 默认=1000
        燃烧期（burn-in）的迭代次数，这些样本将被丢弃
        
    max_expr_depth : int, 默认=5
        表达式树的最大深度
        
    temperature : float, 默认=1.0
        温度参数，控制接受概率
        
    parsimony_coef : float, 默认=0.1
        简洁性系数，惩罚复杂表达式
        
    operators : List[str], 默认=None
        允许的操作符列表
        
    n_chains : int, 默认=3
        MCMC链的数量，多链有助于评估收敛性
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 n_iter: int = 10000,
                 burn_in: int = 1000,
                 max_expr_depth: int = 5,
                 temperature: float = 1.0,
                 parsimony_coef: float = 0.1,
                 operators: Optional[List[str]] = None,
                 n_chains: int = 3,
                 random_state: int = RANDOM_STATE):
        
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.max_expr_depth = max_expr_depth
        self.temperature = temperature
        self.parsimony_coef = parsimony_coef
        self.operators = operators if operators else ['+', '-', '*', '/', 'sqrt', 'square', 'log', 'exp']
        self.n_chains = n_chains
        self.random_state = random_state
        
        # 将操作符分为一元和二元
        self.unary_ops = ['sqrt', 'square', 'log', 'exp', 'sin', 'cos', 'abs', 'neg']
        self.binary_ops = ['+', '-', '*', '/']
        
        # 内部状态
        self._best_expr = None
        self._best_score = float('inf')
        self._feature_names = None
        self._fitted = False
        self._chain_history = []
    
    def _create_random_expr(self, depth=0):
        """创建随机表达式"""
        # 如果达到最大深度或随机决定停止，则创建叶子节点
        if depth >= self.max_expr_depth or (depth > 0 and random.random() < 0.3):
            if random.random() < 0.7:  # 70% 概率选择变量
                return Expression('var', random.choice(self._feature_names))
            else:  # 30% 概率选择常数
                return Expression('const', random.uniform(-5, 5))
        
        # 否则创建操作符节点
        if random.random() < 0.4 and self.unary_ops:  # 40% 概率选择一元操作符
            op = random.choice([op for op in self.unary_ops if op in self.operators])
            left = self._create_random_expr(depth + 1)
            return Expression('unary_op', op, left)
        else:  # 60% 概率选择二元操作符
            op = random.choice([op for op in self.binary_ops if op in self.operators])
            left = self._create_random_expr(depth + 1)
            right = self._create_random_expr(depth + 1)
            return Expression('binary_op', op, left, right)
    
    def _calculate_log_likelihood(self, expr, X, y):
        """计算对数似然"""
        try:
            y_pred = expr.evaluate(X)
            mse = np.mean((y - y_pred) ** 2)
            
            # 避免数值问题
            if np.isnan(mse) or np.isinf(mse):
                return -float('inf')
            
            # 假设噪声为高斯分布
            n = len(y)
            log_lik = -n/2 * np.log(2 * np.pi * mse) - n/2
            
            # 添加表达式复杂度惩罚
            complexity_penalty = self.parsimony_coef * expr.get_complexity()
            
            return log_lik - complexity_penalty
        
        except Exception as e:
            logger.warning(f"评估表达式时出错: {str(e)}")
            return -float('inf')
    
    def _propose_new_expr(self, expr):
        """提出新的表达式（提案分布）"""
        new_expr = expr.copy()
        
        # 随机选择变异类型
        mutation_type = random.choice(['change_node', 'replace_subtree', 'swap_nodes'])
        
        if mutation_type == 'change_node':
            # 改变单个节点
            nodes = new_expr.get_all_nodes()
            if not nodes:
                return self._create_random_expr()
            
            node = random.choice(nodes)
            
            if node.type == 'const':
                # 扰动常数值
                node.value += random.gauss(0, 0.5)
            elif node.type == 'var':
                # 更换变量
                node.value = random.choice(self._feature_names)
            elif node.type == 'unary_op':
                # 更换一元操作符
                node.value = random.choice([op for op in self.unary_ops if op in self.operators])
            elif node.type == 'binary_op':
                # 更换二元操作符
                node.value = random.choice([op for op in self.binary_ops if op in self.operators])
        
        elif mutation_type == 'replace_subtree':
            # 替换子树
            nodes = new_expr.get_all_nodes()
            if not nodes or len(nodes) <= 1:
                return self._create_random_expr()
            
            # 随机选择非根节点
            node = random.choice(nodes[1:])
            
            # 生成新的子树
            depth = node.depth
            new_subtree = self._create_random_expr(depth)
            
            # 替换节点
            found = False
            for parent in nodes:
                if parent.left is node:
                    parent.left = new_subtree
                    found = True
                    break
                elif parent.right is node:
                    parent.right = new_subtree
                    found = True
                    break
            
            if not found:
                # 如果找不到父节点，则生成新的表达式
                return self._create_random_expr()
        
        elif mutation_type == 'swap_nodes':
            # 交换两个节点
            nodes = new_expr.get_all_nodes()
            if len(nodes) <= 2:
                return new_expr
            
            node1, node2 = random.sample(nodes, 2)
            
            # 交换值
            if node1.type == node2.type:
                node1.value, node2.value = node2.value, node1.value
        
        # 更新深度和大小信息
        new_expr.depth = new_expr._calculate_depth()
        new_expr.size = new_expr._calculate_size()
        
        return new_expr
    
    def _run_mcmc_chain(self, X, y, chain_id=0):
        """运行单个MCMC链"""
        random.seed(self.random_state + chain_id)
        np.random.seed(self.random_state + chain_id)
        
        # 初始化
        current_expr = self._create_random_expr()
        current_log_lik = self._calculate_log_likelihood(current_expr, X, y)
        
        best_expr = current_expr
        best_log_lik = current_log_lik
        
        samples = []
        log_liks = []
        accept_count = 0
        
        # 主循环
        for i in range(self.n_iter):
            # 提出新表达式
            proposed_expr = self._propose_new_expr(current_expr)
            proposed_log_lik = self._calculate_log_likelihood(proposed_expr, X, y)
            
            # Metropolis-Hastings 接受概率
            log_accept_ratio = (proposed_log_lik - current_log_lik) / self.temperature
            
            # 决定是否接受
            if log_accept_ratio >= 0 or np.log(random.random()) < log_accept_ratio:
                current_expr = proposed_expr
                current_log_lik = proposed_log_lik
                accept_count += 1
            
            # 更新最佳表达式
            if current_log_lik > best_log_lik:
                best_expr = current_expr.copy()
                best_log_lik = current_log_lik
            
            # 保存样本（燃烧期后）
            if i >= self.burn_in:
                samples.append(current_expr.copy())
                log_liks.append(current_log_lik)
            
            # 输出进度
            if (i + 1) % 1000 == 0:
                logger.info(f"链 {chain_id}: 迭代 {i+1}/{self.n_iter}, 接受率: {accept_count/(i+1):.4f}")
                logger.info(f"当前最佳表达式: {best_expr.to_string()}, 对数似然: {best_log_lik:.4f}")
        
        return {
            'best_expr': best_expr,
            'best_log_lik': best_log_lik,
            'samples': samples,
            'log_liks': log_liks,
            'accept_rate': accept_count / self.n_iter
        }
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        y: Union[np.ndarray, pd.Series, list],
    ):
        """拟合贝叶斯符号回归模型"""
        logger.info("开始贝叶斯符号回归训练...")

        X, y = auto_convert_input(X, y)

        self._feature_names = list(X.columns)
        self._chain_history = []
        
        # 运行多个MCMC链
        results = []
        for i in range(self.n_chains):
            logger.info(f"运行MCMC链 {i+1}/{self.n_chains}...")
            chain_result = self._run_mcmc_chain(X, y, i)
            results.append(chain_result)
            self._chain_history.append({
                'best_expr': chain_result['best_expr'].to_string(),
                'best_log_lik': chain_result['best_log_lik'],
                'accept_rate': chain_result['accept_rate']
            })
        
        # 找到所有链中最好的表达式
        best_idx = np.argmax([r['best_log_lik'] for r in results])
        self._best_expr = results[best_idx]['best_expr']
        self._best_score = results[best_idx]['best_log_lik']

        # 设置训练状态
        self._fitted = True

        # 保存训练数据
        self._train_X = X
        self._train_y = y

        # 计算模型性能
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        logger.info(f"贝叶斯符号回归训练完成，MSE: {mse:.6f}, R²: {r2:.6f}")
        logger.info(f"最佳表达式: {self._best_expr.to_string()}")

        return self
    
    def predict(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """使用训练好的模型进行预测"""
        if not self._fitted:
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
        
        return {
            "expression": self._best_expr.to_string(),
            "log_likelihood": self._best_score,
            "complexity": self._best_expr.get_complexity(),
            "depth": self._best_expr.depth,
            "size": self._best_expr.size,
            "chain_history": self._chain_history
        }
    
    def explain(self):
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
            "configuration": {
                "n_iter": self.n_iter,
                "n_chains": self.n_chains
            },
            "results": {
                "final_model": {
                    "formula_latex": self._best_expr.to_string(),
                    "formula_sympy": self._best_expr.to_string(),
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
