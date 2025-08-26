# -*- coding: utf-8 -*-
"""
Lasso/Ridge/OMP稀疏回归实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, Ridge, OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..config import RANDOM_STATE
from ..utils.data_conversion import auto_convert_input, validate_input_shapes

logger = logging.getLogger(__name__)

class BaseSparseSolver(BaseEstimator, RegressorMixin):
    """
    稀疏回归的基类，提供通用功能
    """
    def __init__(self,
                 poly_degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 normalize: bool = True,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 内部状态
        self.model_ = None
        self.feature_names_ = None
        self.poly_features_ = None
        self.formula_ = None
    
    def _build_feature_names(self, X):
        """构建多项式特征的名称"""
        original_features = X.columns.tolist() if hasattr(X, 'columns') else [f'x{i+1}' for i in range(X.shape[1])]
        
        # 创建多项式特征生成器
        poly = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # 创建一个虚拟数据用于获取特征名称（避免空DataFrame问题）
        if len(X) == 0:
            # 如果输入是空的，创建一个单行虚拟数据
            dummy_data = pd.DataFrame(
                np.zeros((1, len(original_features))), 
                columns=original_features
            )
            poly.fit(dummy_data)
        else:
            poly.fit(X)
            
        powers = poly.powers_
        
        # 构建可读名称
        feature_names = []
        for power in powers:
            name = []
            for i, p in enumerate(power):
                if p > 0:
                    feat = original_features[i]
                    if p == 1:
                        name.append(feat)
                    else:
                        name.append(f"{feat}^{p}")
            
            if not name:  # 常数项
                feature_names.append("1")
            else:
                feature_names.append("*".join(name))
        
        return feature_names
    
    def _build_formula(self, coef, intercept, feature_names):
        """根据系数和特征名构建公式"""
        terms = []
        
        # 添加截距
        if abs(intercept) > 1e-10:
            terms.append(f"{intercept:.6f}")
        
        # 添加特征项
        for i, (name, c) in enumerate(zip(feature_names, coef)):
            if abs(c) > 1e-10:  # 忽略接近零的系数
                sign = "+" if c > 0 and terms else ""
                if name == "1":  # 常数项
                    terms.append(f"{sign} {c:.6f}")
                else:
                    terms.append(f"{sign} {c:.6f}*{name}")
        
        formula = " ".join(terms) if terms else "0"
        return formula
    
    def _get_feature_importance(self, coef, feature_names):
        """计算特征重要性"""
        importance = pd.Series(np.abs(coef), index=feature_names)
        return importance.sort_values(ascending=False)

class LassoRegressor(BaseSparseSolver):
    """
    基于Lasso回归的符号回归实现
    
    Lasso回归通过L1正则化来促进系数稀疏性，适合于发现简洁的数学公式。
    
    参数:
    -----
    alpha : float, 默认=0.1
        L1正则化强度，越大则模型越稀疏
        
    max_iter : int, 默认=1000
        最大迭代次数
        
    tol : float, 默认=1e-4
        收敛容差
        
    poly_degree : int, 默认=2
        多项式特征的最高次数
        
    interaction_only : bool, 默认=False
        是否只包含交互项
        
    include_bias : bool, 默认=True
        是否包含偏置项
        
    normalize : bool, 默认=True
        是否标准化特征
        
    cv : int, 默认=5
        交叉验证折数，用于自动选择alpha
        
    n_jobs : int, 默认=1
        并行作业数
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 alpha: float = 0.1,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 poly_degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 normalize: bool = True,
                 cv: int = 5,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        super().__init__(
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            normalize=normalize,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, list],
            feature_names: Optional[List[str]] = None):
        """
        拟合Lasso回归模型
        
        参数:
        -----
        X : np.ndarray or pd.DataFrame
            输入特征，支持numpy数组和pandas DataFrame
            
        y : np.ndarray, pd.Series or list
            目标变量，支持numpy数组、pandas Series和列表
            
        feature_names : List[str], 可选
            特征名称列表，仅在输入为numpy数组时需要
        """
        logger.info("开始训练Lasso回归模型...")
        
        # 验证输入形状并转换为pandas格式
        validate_input_shapes(X, y)
        X, y = auto_convert_input(X, y, feature_names)
        self._train_X = X
        self._train_y = y
        
        # 保存特征名
        self.feature_names_ = X.columns.tolist()
        
        # 创建多项式特征
        self.poly_features_ = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # 创建标准化器
        scaler = StandardScaler() if self.normalize else None
        
        # 构建管道
        steps = []
        steps.append(('poly', self.poly_features_))
        if scaler:
            steps.append(('scaler', scaler))
        
        if self.cv > 1:
            # 使用交叉验证自动选择alpha
            from sklearn.linear_model import LassoCV
            lasso = LassoCV(
                alphas=np.logspace(-4, 1, 20),
                cv=self.cv,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            steps.append(('lasso', lasso))
            
            self.pipeline_ = Pipeline(steps)
            self.pipeline_.fit(X, y)
            
            self.model_ = self.pipeline_.named_steps['lasso']
            self.alpha_ = self.model_.alpha_
            logger.info(f"通过交叉验证选择的alpha: {self.alpha_:.6f}")
        else:
            # 使用固定alpha
            lasso = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            steps.append(('lasso', lasso))
            
            self.pipeline_ = Pipeline(steps)
            self.pipeline_.fit(X, y)
            
            self.model_ = self.pipeline_.named_steps['lasso']
            self.alpha_ = self.alpha
        
        # 获取多项式特征名
        feature_names = self._build_feature_names(X)
        
        # 构建公式
        self.formula_ = self._build_formula(
            self.model_.coef_, 
            self.model_.intercept_, 
            feature_names
        )
        
        # 计算性能
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        n_nonzero = np.sum(self.model_.coef_ != 0)
        
        logger.info(f"Lasso回归模型训练完成，MSE: {mse:.6f}, R2: {r2:.6f}")
        logger.info(f"非零系数数量: {n_nonzero}/{len(self.model_.coef_)}")
        logger.info(f"公式: {self.formula_}")

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
        if self.pipeline_ is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        # 验证输入形状并转换为pandas格式
        from ..utils.data_conversion import ensure_pandas_dataframe
        X_df = ensure_pandas_dataframe(X, feature_names)
        
        # 增强的输入验证和错误处理
        logger.info(f"[LASSO预测] 输入数据形状: {X_df.shape}")
        logger.info(f"[LASSO预测] 输入数据列: {list(X_df.columns)}")
        
        if X_df.empty:
            raise ValueError("预测数据不能为空")
        
        # 如果输入是0行数据，直接返回空数组
        if len(X_df) == 0:
            logger.warning("[LASSO预测] 输入数据为空，返回空预测结果")
            return np.array([])
        
        try:
            # 调试：打印管道中每个步骤的输出
            logger.info("[LASSO预测] 开始预测过程...")
            
            # 检查多项式特征转换
            poly_step = self.pipeline_.named_steps['poly']
            logger.info(f"[LASSO预测] 多项式特征转换器: {poly_step}")
            
            # 手动执行多项式特征转换以检查问题
            try:
                X_poly = poly_step.transform(X_df)
                logger.info(f"[LASSO预测] 多项式特征转换后形状: {X_poly.shape}")
                
                if X_poly.shape[0] == 0:
                    logger.error(f"[LASSO预测] 多项式转换产生了空数组，原始输入: {X_df.shape}")
                    logger.error(f"[LASSO预测] 原始数据样本: {X_df.head(3)}")
                    raise ValueError("多项式特征转换产生了空数组")
                    
            except Exception as poly_err:
                logger.error(f"[LASSO预测] 多项式特征转换失败: {str(poly_err)}")
                logger.error(f"[LASSO预测] 输入数据信息: shape={X_df.shape}, dtype={X_df.dtypes}")
                logger.error(f"[LASSO预测] 输入数据内容: {X_df.head()}")
                raise
            
            # 执行完整预测
            result = self.pipeline_.predict(X_df)
            logger.info(f"[LASSO预测] 预测完成，结果形状: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"[LASSO预测] 预测时发生错误: {str(e)}")
            logger.error(f"[LASSO预测] 输入数据形状: {X_df.shape}")
            logger.error(f"[LASSO预测] 输入数据列: {list(X_df.columns)}")
            logger.error(f"[LASSO预测] 管道步骤: {[step for step, _ in self.pipeline_.steps]}")
            
            # 尝试诊断问题
            if hasattr(self.pipeline_, 'named_steps'):
                for step_name, step in self.pipeline_.named_steps.items():
                    logger.error(f"[LASSO预测] 步骤 {step_name}: {step}")
            
            raise
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        -----
        Dict
            模型信息字典
        """
        if self.model_ is None:
            return {"status": "模型尚未训练"}
        
        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        importance = self._get_feature_importance(self.model_.coef_, feature_names)
        
        return {
            "formula": self.formula_,
            "alpha": self.alpha_,
            "nonzero_terms": int(np.sum(self.model_.coef_ != 0)),
            "total_terms": len(self.model_.coef_),
            "top_features": importance.head(5).to_dict(),
            "intercept": self.model_.intercept_
        }
    
    def explain(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.model_ is None:
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

        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        nonzero_terms = {
            name: coef for name, coef in zip(feature_names, self.model_.coef_)
            if abs(coef) > 1e-10
        }

        report = {
            "configuration": {
                "alpha": self.alpha_,
                "poly_degree": self.poly_degree
            },
            "results": {
                "final_model": {
                    "formula_latex": self.formula_,
                    "formula_sympy": self.formula_,
                    "intercept": self.model_.intercept_,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": int(np.sum(self.model_.coef_ != 0))
            }
        }

        return SissoReport(report)

class RidgeRegressor(BaseSparseSolver):
    """
    基于Ridge回归的符号回归实现
    
    Ridge回归通过L2正则化来稳定系数，适合于特征间存在多重共线性的情况。
    
    参数:
    -----
    alpha : float, 默认=1.0
        L2正则化强度
        
    max_iter : int, 默认=1000
        最大迭代次数
        
    tol : float, 默认=1e-4
        收敛容差
        
    poly_degree : int, 默认=2
        多项式特征的最高次数
        
    interaction_only : bool, 默认=False
        是否只包含交互项
        
    include_bias : bool, 默认=True
        是否包含偏置项
        
    normalize : bool, 默认=True
        是否标准化特征
        
    cv : int, 默认=5
        交叉验证折数，用于自动选择alpha
        
    n_jobs : int, 默认=1
        并行作业数
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 tol: float = 1e-4,
                 poly_degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 normalize: bool = True,
                 cv: int = 5,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        super().__init__(
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            normalize=normalize,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        拟合Ridge回归模型
        
        参数:
        -----
        X : pd.DataFrame
            输入特征
            
        y : pd.Series
            目标变量
        """
        logger.info("开始训练Ridge回归模型...")
        
        # 保存特征名
        self.feature_names_ = X.columns.tolist()
        
        # 创建多项式特征
        self.poly_features_ = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # 创建标准化器
        scaler = StandardScaler() if self.normalize else None
        
        # 构建管道
        steps = []
        steps.append(('poly', self.poly_features_))
        if scaler:
            steps.append(('scaler', scaler))
        
        if self.cv > 1:
            # 使用交叉验证自动选择alpha
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(
                alphas=np.logspace(-4, 4, 20),
                cv=self.cv,
                scoring='neg_mean_squared_error'
            )
            steps.append(('ridge', ridge))
            
            self.pipeline_ = Pipeline(steps)
            self.pipeline_.fit(X, y)
            
            self.model_ = self.pipeline_.named_steps['ridge']
            self.alpha_ = self.model_.alpha_
            logger.info(f"通过交叉验证选择的alpha: {self.alpha_:.6f}")
        else:
            # 使用固定alpha
            ridge = Ridge(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            steps.append(('ridge', ridge))
            
            self.pipeline_ = Pipeline(steps)
            self.pipeline_.fit(X, y)
            
            self.model_ = self.pipeline_.named_steps['ridge']
            self.alpha_ = self.alpha
        
        # 获取多项式特征名
        feature_names = self._build_feature_names(X)
        
        # 构建公式
        self.formula_ = self._build_formula(
            self.model_.coef_, 
            self.model_.intercept_, 
            feature_names
        )
        
        # 计算性能
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Ridge通常不会产生精确的零系数，但我们可以设置一个阈值
        small_coef = np.sum(np.abs(self.model_.coef_) < 1e-4)
        
        logger.info(f"Ridge回归模型训练完成，MSE: {mse:.6f}, R2: {r2:.6f}")
        logger.info(f"近似零系数数量: {small_coef}/{len(self.model_.coef_)}")
        logger.info(f"公式: {self.formula_}")

        self._train_X = X
        self._train_y = y

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
        if self.pipeline_ is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        return self.pipeline_.predict(X)
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        -----
        Dict
            模型信息字典
        """
        if self.model_ is None:
            return {"status": "模型尚未训练"}
        
        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        importance = self._get_feature_importance(self.model_.coef_, feature_names)
        
        # 使用阈值定义"重要"特征
        significant_terms = np.sum(np.abs(self.model_.coef_) > 1e-4)
        
        return {
            "formula": self.formula_,
            "alpha": self.alpha_,
            "significant_terms": int(significant_terms),
            "total_terms": len(self.model_.coef_),
            "top_features": importance.head(5).to_dict(),
            "intercept": self.model_.intercept_
        }
    
    def explain(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.model_ is None:
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

        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        significant_terms = {name: coef for name, coef in zip(feature_names, self.model_.coef_)
                             if abs(coef) > 1e-4}

        report = {
            "configuration": {
                "alpha": self.alpha_,
                "poly_degree": self.poly_degree
            },
            "results": {
                "final_model": {
                    "formula_latex": self.formula_,
                    "formula_sympy": self.formula_,
                    "intercept": self.model_.intercept_,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": len(significant_terms)
            }
        }

        return SissoReport(report)

class OMPRegressor(BaseSparseSolver):
    """
    基于正交匹配追踪 (OMP) 的符号回归实现
    
    OMP是一种贪婪算法，它通过迭代地选择与残差最相关的特征来构建稀疏模型。
    
    参数:
    -----
    n_nonzero_coefs : int, 默认=None
        非零系数的最大数量，如果为None则使用tol参数
        
    tol : float, 默认=None
        残差的容差，如果n_nonzero_coefs为None则使用该参数
        
    poly_degree : int, 默认=2
        多项式特征的最高次数
        
    interaction_only : bool, 默认=False
        是否只包含交互项
        
    include_bias : bool, 默认=True
        是否包含偏置项
        
    normalize : bool, 默认=True
        是否标准化特征
        
    n_jobs : int, 默认=1
        并行作业数
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 n_nonzero_coefs: Optional[int] = None,
                 tol: Optional[float] = None,
                 poly_degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 normalize: bool = True,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        super().__init__(
            poly_degree=poly_degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            normalize=normalize,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        # 至少需要指定一个参数
        if n_nonzero_coefs is None and tol is None:
            n_nonzero_coefs = 5  # 默认值
            
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        拟合OMP回归模型
        
        参数:
        -----
        X : pd.DataFrame
            输入特征
            
        y : pd.Series
            目标变量
        """
        logger.info("开始训练OMP回归模型...")
        
        # 保存特征名
        self.feature_names_ = X.columns.tolist()
        
        # 创建多项式特征
        self.poly_features_ = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # 创建标准化器
        scaler = StandardScaler() if self.normalize else None
        
        # 构建管道
        steps = []
        steps.append(('poly', self.poly_features_))
        if scaler:
            steps.append(('scaler', scaler))
        
        # 创建OMP模型
        # sklearn >=1.2 移除了 normalize 参数，标准化已经在管道中完成
        omp = OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.n_nonzero_coefs,
            tol=self.tol
        )
        steps.append(('omp', omp))
        
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X, y)
        
        self.model_ = self.pipeline_.named_steps['omp']
        
        # 获取多项式特征名
        feature_names = self._build_feature_names(X)
        
        # 构建公式
        self.formula_ = self._build_formula(
            self.model_.coef_, 
            self.model_.intercept_, 
            feature_names
        )
        
        # 计算性能
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        n_nonzero = np.sum(self.model_.coef_ != 0)
        
        logger.info(f"OMP回归模型训练完成，MSE: {mse:.6f}, R2: {r2:.6f}")
        logger.info(f"非零系数数量: {n_nonzero}/{len(self.model_.coef_)}")
        logger.info(f"公式: {self.formula_}")

        self._train_X = X
        self._train_y = y

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
        if self.pipeline_ is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        return self.pipeline_.predict(X)
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        -----
        Dict
            模型信息字典
        """
        if self.model_ is None:
            return {"status": "模型尚未训练"}
        
        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        importance = self._get_feature_importance(self.model_.coef_, feature_names)
        
        return {
            "formula": self.formula_,
            "n_nonzero_coefs": int(np.sum(self.model_.coef_ != 0)),
            "total_terms": len(self.model_.coef_),
            "top_features": importance.head(5).to_dict(),
            "intercept": self.model_.intercept_
        }
    
    def explain(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.model_ is None:
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

        feature_names = self._build_feature_names(pd.DataFrame(columns=self.feature_names_))
        nonzero_terms = {name: coef for name, coef in zip(feature_names, self.model_.coef_)
                         if abs(coef) > 1e-10}

        report = {
            "configuration": {
                "n_nonzero_coefs": self.n_nonzero_coefs,
                "tol": self.tol,
                "poly_degree": self.poly_degree
            },
            "results": {
                "final_model": {
                    "formula_latex": self.formula_,
                    "formula_sympy": self.formula_,
                    "intercept": self.model_.intercept_,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": int(np.sum(self.model_.coef_ != 0))
            }
        }

        return SissoReport(report)
