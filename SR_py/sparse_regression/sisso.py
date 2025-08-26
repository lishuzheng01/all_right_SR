# -*- coding: utf-8 -*-
"""
SISSO (大规模符号特征生成 + 稀疏筛选) 实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import logging
from sklearn.base import BaseEstimator, RegressorMixin

from ..dsl.expr import Var, Expr
from ..dsl.dimension import Dimension
from ..gen.generator import FeatureGenerator
from ..gen.evaluator import FeatureEvaluator
from ..sis.screening import SIS
from ..sis.so import SparsifyingOperator
from ..config import DEFAULT_OPERATORS, RANDOM_STATE
from ..model.report import build_report
from ..utils.data_conversion import auto_convert_input, validate_input_shapes

logger = logging.getLogger(__name__)

class SISSORegressor(BaseEstimator, RegressorMixin):
    """
    升级版SISSO (大规模符号特征生成 + 稀疏筛选) 回归器
    
    SISSO是一种用于从数据中发现物理规律和数学关系的方法，
    它结合了大规模特征空间生成和稀疏特征选择。
    
    参数:
    -----
    K : int, 默认=3
        特征复杂度层级
        
    operators : List[str], 默认=None
        用于构建表达式的操作符列表
        
    sis_screener : str, 默认='pearson'
        特征筛选方法，可选值: 'pearson', 'mutual_info', 'f_regression', 'rfe', 'variance'
        
    sis_topk : int, 默认=2000
        每层保留的特征数
        
    so_solver : str, 默认='omp'
        稀疏求解器，可选值: 'omp', 'lasso', 'elasticnet'
        
    so_max_terms : int, 默认=3
        最终模型的最大项数
        
    cv : int, 默认=5
        交叉验证折数
        
    dimensional_check : bool, 默认=False
        是否检查物理量纲一致性
        
    n_jobs : int, 默认=-1
        并行计算的作业数，-1表示使用所有处理器

    random_state : int, 默认=42
        随机种子

    示例:
    -----
    >>> from SR_py.sparse_regression.sisso import SISSORegressor
    >>> model = SISSORegressor(K=2)
    >>> model.fit(X, y)
    >>> print(model.explain())
    """
    def __init__(self,
                 K: int = 3,
                 operators: List[Union[str, Callable, Tuple]] = None,
                 sis_screener: str = 'pearson',
                 sis_topk: int = 2000,
                 so_solver: str = 'omp',
                 so_max_terms: int = 3,
                 cv: int = 5,
                 dimensional_check: bool = False,
                 n_jobs: int = -1,
                 random_state: int = RANDOM_STATE):
        
        self.K = K
        self.operators = operators if operators is not None else DEFAULT_OPERATORS
        self.sis_screener = sis_screener
        self.sis_topk = sis_topk
        self.so_solver = so_solver
        self.so_max_terms = so_max_terms
        self.cv = cv
        self.dimensional_check = dimensional_check
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 内部组件
        self.generator = None
        self.evaluator = None
        self.screener = None
        self.solver = None
        
        # 结果
        self.final_model_info = {}
        self.feature_space_ = []
        self.screened_features_ = pd.DataFrame()

    def _initialize_components(self):
        """初始化所有内部组件"""
        # 初始化特征生成器
        self.generator = FeatureGenerator(
            operators=self.operators,
            max_complexity=self.K * 3,  # 大致对应深度
            dimensional_check=self.dimensional_check
        )
        
        # 初始化特征评估器
        self.evaluator = FeatureEvaluator()
        
        # 初始化SIS筛选器
        self.screener = SIS(
            screener=self.sis_screener, 
            top_k=self.sis_topk
        )
        
        # 初始化稀疏求解器
        self.solver = SparsifyingOperator(
            solver=self.so_solver, 
            max_terms=self.so_max_terms, 
            cv=self.cv
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, list], 
            feature_dimensions: Optional[Dict[str, Dimension]] = None,
            target_dimension: Optional[Dimension] = None,
            feature_names: Optional[List[str]] = None):
        """
        拟合SISSO模型
        
        参数:
        -----
        X : np.ndarray or pd.DataFrame
            输入特征，支持numpy数组和pandas DataFrame
            
        y : np.ndarray, pd.Series or list
            目标变量，支持numpy数组、pandas Series和列表
            
        feature_dimensions : Dict[str, Dimension], 可选
            特征的物理量纲字典
            
        target_dimension : Dimension, 可选
            目标变量的物理量纲
            
        feature_names : List[str], 可选
            特征名称列表，仅在输入为numpy数组时需要
        """
        logger.info("开始SISSO模型训练...")
        
        # 验证输入形状
        validate_input_shapes(X, y)
        
        # 自动转换输入为pandas格式
        X, y = auto_convert_input(X, y, feature_names)
        
        # 存储原始数据
        self.X_original_ = X
        self.y_original_ = y
        
        # 初始化组件
        self._initialize_components()
        
        # 1. 生成特征空间
        if self.dimensional_check and feature_dimensions is None:
            raise ValueError("启用量纲检查时必须提供feature_dimensions")
        
        # 创建初始变量
        initial_features = [
            Var(name, dimension=feature_dimensions.get(name))
            if feature_dimensions else Var(name)
            for name in X.columns
        ]
        
        # 生成分层特征
        logger.info(f"正在生成特征空间，复杂度层级 K={self.K}...")
        feature_layers = self.generator.generate(initial_features, self.K)
        self.feature_space_ = [expr for layer in feature_layers for expr in layer]
        
        # 量纲筛选
        if self.dimensional_check and target_dimension:
            logger.info(f"根据目标量纲筛选特征: {target_dimension.to_string()}")
            original_count = len(self.feature_space_)
            self.feature_space_ = [
                f for f in self.feature_space_ if f.dimension == target_dimension
            ]
            logger.info(f"保留了 {len(self.feature_space_)}/{original_count} 个量纲正确的特征")
        
        # 2. 评估特征
        logger.info(f"评估 {len(self.feature_space_)} 个特征...")
        evaluated_features_df, valid_features = self.evaluator.evaluate(self.feature_space_, X)
        
        # 3. 特征筛选 (SIS)
        logger.info(f"使用 {self.sis_screener} 方法筛选特征...")
        self.screened_features_ = self.screener.screen(evaluated_features_df, y)
        
        # 4. 构建稀疏模型 (SO)
        logger.info(f"使用 {self.so_solver} 求解器构建稀疏模型...")
        self.solver.fit(self.screened_features_, y)
        self.final_model_info = self.solver.get_model_info()
        
        # 保存训练数据
        self._train_X = X.copy()
        self._train_y = y.copy()
        
        logger.info(f"SISSO模型训练完成，选择了 {len(self.final_model_info.get('selected_features', []))} 个特征")
        
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame], 
                feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
        -----
        X : np.ndarray or pd.DataFrame
            输入特征，支持numpy数组和pandas DataFrame
            
        feature_names : List[str], 可选
            特征名称列表，仅在输入为numpy数组时需要
            
        返回:
        -----
        np.ndarray
            预测值
        """
        if not self.solver or not self.final_model_info:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        # 验证输入形状并转换为pandas格式
        from ..utils.data_conversion import ensure_pandas_dataframe
        X_df = ensure_pandas_dataframe(X, feature_names)
        
        # 如果特征名称不匹配训练时的特征，尝试对齐
        if hasattr(self, 'X_original_') and not X_df.columns.equals(self.X_original_.columns):
            if len(X_df.columns) == len(self.X_original_.columns):
                X_df.columns = self.X_original_.columns
            else:
                raise ValueError(f"特征数量不匹配。训练时: {len(self.X_original_.columns)}, 预测时: {len(X_df.columns)}")
        
        # 获取选定的特征名
        selected_feature_names = self.final_model_info.get('selected_features', [])
        
        # 找到对应的表达式对象
        selected_exprs = [expr for expr in self.feature_space_ 
                           if expr.get_signature() in selected_feature_names]
        
        # 评估特征
        X_eval, _ = self.evaluator.evaluate(selected_exprs, X_df)
        
        # 预测
        return self.solver.predict(X_eval)

    def _build_report(self) -> Dict:
        """
        返回最终模型的解释信息

        返回:
        -----
        Dict
            包含模型信息的字典
        """
        return build_report(self)

    @property
    def explain(self) -> Dict:
        return self._build_report()

    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        -----
        Dict
            包含模型信息的字典
        """
        if not self.final_model_info:
            return {"status": "模型尚未训练"}
        
        # 获取选中的表达式
        selected_features = self.final_model_info.get('selected_features', [])
        selected_exprs = [expr for expr in self.feature_space_ 
                          if expr.get_signature() in selected_features]
        
        # 系数
        coefficients = self.final_model_info.get('coefficients', {})
        
        # 构建公式字符串
        formula_parts = []
        intercept = self.final_model_info.get('intercept', 0)
        
        if abs(intercept) > 1e-10:
            formula_parts.append(f"{intercept:.6f}")
        
        for expr in selected_exprs:
            coef = coefficients.get(expr.get_signature(), 0)
            if abs(coef) > 1e-10:
                sign = "+" if coef > 0 and formula_parts else ""
                formula_parts.append(f"{sign} {coef:.6f} * {str(expr)}")
        
        formula = " ".join(formula_parts) if formula_parts else "0"
        
        return {
            **self.final_model_info,
            "formula": formula,
            "expressions": [str(expr) for expr in selected_exprs],
            "method": f"SISSO-{self.K}-{self.sis_screener}-{self.so_solver}"
        }
