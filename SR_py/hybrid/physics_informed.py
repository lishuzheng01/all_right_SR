# -*- coding: utf-8 -*-
"""
物理约束的符号回归方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class PhysicsInformedSymbolicRegression(BaseEstimator, RegressorMixin):
    """Physics-informed symbolic regression.

    Generates expressions that respect user-specified physical constraints and
    optional dimensional analysis checks to ensure the resulting formulas are
    physically plausible.

    Parameters
    ----------
    K : int, default=2
        Feature complexity level used in the SISSO-style generation stage.
    physical_constraints : list[str], default=['conservation_laws']
        List of physical laws to enforce during feature selection.
    dimensional_analysis : bool, default=True
        Whether to enforce dimensional consistency.
    constraint_weight : float, default=0.1
        Weighting factor for constraint violation penalties.
    sis_topk : int, default=100
        Number of top features retained after screening.
    so_max_terms : int, default=3
        Maximum terms allowed in the final sparse model.
    operators : list[str] or None, default=None
        Operators available for feature construction. ``None`` uses a default
        operator set.
    max_depth : int, default=5
        Maximum depth of expression trees during symbolic optimisation.
    population_size : int, default=50
        Population size for evolutionary search.
    generations : int, default=30
        Number of generations in the symbolic search stage.
    random_state : int, default=42
        Seed for reproducibility.

    Examples
    --------
    >>> from SR_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
    >>> model = PhysicsInformedSymbolicRegression()
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    
    def __init__(self,
                 K=2,
                 physical_constraints=['conservation_laws'],
                 dimensional_analysis=True,
                 constraint_weight=0.1,
                 sis_topk=100,
                 so_max_terms=3,
                 operators=None,
                 max_depth=5,
                 population_size=50,
                 generations=30,
                 random_state=42):
        
        self.K = K
        self.physical_constraints = physical_constraints
        self.dimensional_analysis = dimensional_analysis
        self.constraint_weight = constraint_weight
        self.sis_topk = sis_topk
        self.so_max_terms = so_max_terms
        self.operators = operators
        self.max_depth = max_depth
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state
        
        self.best_expression_ = None
        self.best_score_ = float('inf')
        self.dimensional_violations_ = []
        self.constraint_violations_ = []
        
    def fit(self, X, y, feature_names=None, feature_dimensions=None, target_dimension=None):
        """
        训练物理约束符号回归模型
        """
        logger.info("开始物理约束符号回归训练...")
        
        np.random.seed(self.random_state)
        
        self.feature_names_ = feature_names or [f'x{i+1}' for i in range(X.shape[1])]
        self.feature_dimensions_ = feature_dimensions
        self.target_dimension_ = target_dimension
        self.n_features_ = X.shape[1]
        
        # 生成候选特征
        logger.info("生成物理约束候选特征...")
        candidate_features = self._generate_physics_informed_features(X)
        
        # 特征筛选
        logger.info("基于物理约束的特征筛选...")
        selected_features = self._physics_informed_screening(candidate_features, X, y)
        
        # 符号回归
        logger.info("符号组合优化...")
        self._symbolic_regression_with_constraints(selected_features, X, y)
        
        logger.info(f"训练完成，最佳表达式: {self.best_expression_}")
        self._train_X = X
        self._train_y = y
        return self
    
    def predict(self, X):
        """预测"""
        if self.best_expression_ is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        return self._evaluate_expression(self.best_expression_, X)
    
    def _generate_physics_informed_features(self, X):
        """生成考虑物理约束的候选特征"""
        features = []
        
        # 基础特征
        for i, name in enumerate(self.feature_names_):
            features.append({
                'expression': name,
                'dimension': self.feature_dimensions_[name] if self.feature_dimensions_ else None,
                'complexity': 1
            })
        
        # 生成物理合理的组合特征
        for i in range(len(self.feature_names_)):
            for j in range(i, len(self.feature_names_)):
                name1, name2 = self.feature_names_[i], self.feature_names_[j]
                
                # 检查维度兼容性
                if self.dimensional_analysis and self.feature_dimensions_:
                    dim1 = self.feature_dimensions_[name1]
                    dim2 = self.feature_dimensions_[name2]
                    
                    # 乘法组合
                    mult_dim = self._add_dimensions(dim1, dim2)
                    if self._is_dimensionally_valid(mult_dim):
                        features.append({
                            'expression': f"({name1} * {name2})",
                            'dimension': mult_dim,
                            'complexity': 2
                        })
                    
                    # 除法组合
                    if not self._is_zero_dimension(dim2):
                        div_dim = self._subtract_dimensions(dim1, dim2)
                        if self._is_dimensionally_valid(div_dim):
                            features.append({
                                'expression': f"({name1} / {name2})",
                                'dimension': div_dim,
                                'complexity': 2
                            })
                
                # 加法和减法（需要相同维度）
                if self.dimensional_analysis and self.feature_dimensions_:
                    if self._dimensions_equal(
                        self.feature_dimensions_[name1], 
                        self.feature_dimensions_[name2]
                    ):
                        features.append({
                            'expression': f"({name1} + {name2})",
                            'dimension': self.feature_dimensions_[name1],
                            'complexity': 2
                        })
                        features.append({
                            'expression': f"({name1} - {name2})",
                            'dimension': self.feature_dimensions_[name1],
                            'complexity': 2
                        })
        
        # 生成物理函数特征（如平方、开方等）
        for name in self.feature_names_:
            if self.feature_dimensions_:
                dim = self.feature_dimensions_[name]
                
                # 平方
                square_dim = self._multiply_dimension_by_scalar(dim, 2)
                if self._is_dimensionally_valid(square_dim):
                    features.append({
                        'expression': f"({name} ** 2)",
                        'dimension': square_dim,
                        'complexity': 2
                    })
                
                # 开方
                sqrt_dim = self._multiply_dimension_by_scalar(dim, 0.5)
                if self._is_dimensionally_valid(sqrt_dim):
                    features.append({
                        'expression': f"sqrt(abs({name}))",
                        'dimension': sqrt_dim,
                        'complexity': 3
                    })
        
        return features
    
    def _physics_informed_screening(self, features, X, y):
        """基于物理约束的特征筛选"""
        valid_features = []
        
        for feature in features:
            # 检查维度约束
            if self.dimensional_analysis and self.target_dimension_:
                if not self._is_target_dimension_compatible(feature['dimension']):
                    self.dimensional_violations_.append(feature['expression'])
                    continue
            
            # 检查物理约束
            if not self._satisfies_physical_constraints(feature):
                self.constraint_violations_.append(feature['expression'])
                continue
            
            # 计算特征值
            try:
                feature_values = self._evaluate_expression(feature['expression'], X)
                if np.any(np.isnan(feature_values)) or np.any(np.isinf(feature_values)):
                    continue
                
                # 计算与目标的相关性
                correlation = np.abs(np.corrcoef(feature_values, y)[0, 1])
                feature['correlation'] = correlation
                
                valid_features.append(feature)
                
            except:
                continue
        
        # 按相关性排序并选择Top-K
        valid_features.sort(key=lambda x: x['correlation'], reverse=True)
        return valid_features[:self.sis_topk]
    
    def _symbolic_regression_with_constraints(self, features, X, y):
        """带约束的符号回归"""
        # 简化的符号回归实现
        best_combination = None
        best_score = float('inf')
        
        # 尝试不同的特征组合
        for combination_size in range(1, min(self.so_max_terms + 1, len(features) + 1)):
            # 生成所有可能的组合
            from itertools import combinations
            
            for feature_combo in combinations(features, combination_size):
                # 构建线性组合
                expression_parts = []
                for i, feature in enumerate(feature_combo):
                    expression_parts.append(f"c{i} * ({feature['expression']})")
                
                if expression_parts:
                    expression = " + ".join(expression_parts) + " + c_intercept"
                else:
                    continue
                
                # 评估组合
                score = self._evaluate_combination_with_constraints(
                    expression, feature_combo, X, y
                )
                
                if score < best_score:
                    best_score = score
                    best_combination = (expression, feature_combo)
        
        if best_combination:
            self.best_expression_ = best_combination[0]
            self.best_score_ = best_score
        else:
            # 回退到简单线性组合
            self.best_expression_ = " + ".join([f"c{i} * {name}" for i, name in enumerate(self.feature_names_)]) + " + c_intercept"
    
    def _evaluate_combination_with_constraints(self, expression, features, X, y):
        """评估带约束的特征组合"""
        try:
            # 使用最小二乘拟合系数
            feature_matrix = []
            for feature in features:
                feature_values = self._evaluate_expression(feature['expression'], X)
                feature_matrix.append(feature_values)
            
            if feature_matrix:
                feature_matrix = np.column_stack(feature_matrix)
                # 添加截距项
                feature_matrix = np.column_stack([feature_matrix, np.ones(X.shape[0])])
                
                # 最小二乘求解
                coefficients = np.linalg.lstsq(feature_matrix, y, rcond=None)[0]
                
                # 预测
                y_pred = feature_matrix @ coefficients
                
                # 计算基础误差
                mse = mean_squared_error(y, y_pred)
                
                # 添加约束惩罚
                constraint_penalty = self._calculate_constraint_penalty(features, coefficients)
                
                return mse + self.constraint_weight * constraint_penalty
            else:
                return 1e6
                
        except:
            return 1e6
    
    def _calculate_constraint_penalty(self, features, coefficients):
        """计算约束惩罚项"""
        penalty = 0
        
        # 守恒定律约束
        if 'conservation_laws' in self.physical_constraints:
            # 简化的能量守恒检查：系数不应过大
            penalty += np.sum(np.abs(coefficients) ** 2) * 0.01
        
        # 维度一致性约束
        if self.dimensional_analysis and self.target_dimension_:
            for feature, coeff in zip(features, coefficients[:-1]):  # 排除截距
                if feature['dimension'] != self.target_dimension_:
                    penalty += abs(coeff) * 0.1
        
        return penalty
    
    def _evaluate_expression(self, expression, X):
        """评估表达式"""
        try:
            # 创建局部环境
            local_env = {}
            
            # 确保X是numpy数组或可以正确索引
            if isinstance(X, pd.DataFrame):
                for i, name in enumerate(self.feature_names_):
                    local_env[name] = X.iloc[:, i].values
            else:
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
            
            # 添加系数（简化处理）
            for i in range(10):  # 支持最多10个系数
                local_env[f'c{i}'] = 1.0
            local_env['c_intercept'] = 0.0
            
            result = eval(expression, {"__builtins__": {}}, local_env)
            
            if np.isscalar(result):
                result = np.full(X.shape[0], result)
            
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        except:
            return np.zeros(X.shape[0])
    
    # 维度分析辅助方法
    def _add_dimensions(self, dim1, dim2):
        """维度相加（用于乘法）"""
        if dim1 is None or dim2 is None:
            return None
        return type(dim1)(dim1.vector + dim2.vector)
    
    def _subtract_dimensions(self, dim1, dim2):
        """维度相减（用于除法）"""
        if dim1 is None or dim2 is None:
            return None
        return type(dim1)(dim1.vector - dim2.vector)
    
    def _multiply_dimension_by_scalar(self, dim, scalar):
        """维度乘以标量"""
        if dim is None:
            return None
        return type(dim)(dim.vector * scalar)
    
    def _dimensions_equal(self, dim1, dim2):
        """检查两个维度是否相等"""
        if dim1 is None or dim2 is None:
            return False
        return np.array_equal(dim1.vector, dim2.vector)
    
    def _is_zero_dimension(self, dim):
        """检查是否为零维度（无量纲）"""
        if dim is None:
            return False
        return np.all(dim.vector == 0)
    
    def _is_dimensionally_valid(self, dim):
        """检查维度是否有效"""
        if dim is None:
            return True
        return np.all(np.abs(dim.vector) <= 10)  # 避免极端维度
    
    def _is_target_dimension_compatible(self, feature_dim):
        """检查特征维度是否与目标维度兼容"""
        if feature_dim is None or self.target_dimension_ is None:
            return True
        return self._dimensions_equal(feature_dim, self.target_dimension_)
    
    def _satisfies_physical_constraints(self, feature):
        """检查是否满足物理约束"""
        # 简化的物理约束检查
        expression = feature['expression']
        
        # 避免不合理的组合
        if 'log(' in expression and 'exp(' in expression:
            return False  # 避免log(exp(...))这样的组合
        
        # 避免过度复杂的表达式
        if feature['complexity'] > 5:
            return False
        
        return True
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'formula': self.best_expression_,
            'best_score': self.best_score_,
            'method': 'Physics-Informed Symbolic Regression',
            'physical_constraints': self.physical_constraints,
            'dimensional_analysis': self.dimensional_analysis,
            'dimensional_violations': len(self.dimensional_violations_),
            'constraint_violations': len(self.constraint_violations_)
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
            "title": "Physics-Informed Symbolic Regression 分析报告",
            "configuration": {
                "K": self.K,
                "physical_constraints": self.physical_constraints,
                "dimensional_analysis": self.dimensional_analysis
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
