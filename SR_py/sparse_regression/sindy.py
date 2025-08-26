# -*- coding: utf-8 -*-
"""
稀疏识别非线性动力学 (SINDy) 实现
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import warnings
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..config import RANDOM_STATE
from ..utils.data_conversion import auto_convert_input, ensure_pandas_dataframe

logger = logging.getLogger(__name__)

class SINDyRegressor(BaseEstimator, RegressorMixin):
    """
    稀疏识别非线性动力学 (SINDy) 回归器
    
    SINDy是一种用于发现动态系统中的非线性关系的方法，它通过生成一组候选特征函数，
    然后使用稀疏回归来识别系统的动力学方程。
    
    参数:
    -----
    poly_degree : int, 默认=3
        多项式特征的最高次数
        
    include_trig : bool, 默认=False
        是否包括三角函数特征
        
    include_exp : bool, 默认=False
        是否包括指数和对数特征
        
    feature_library : List[str], 默认=None
        自定义特征库，可选项: 'constant', 'linear', 'poly', 'sin', 'cos', 'exp', 'log'
        
    alpha : float, 默认=0.1
        稀疏回归的正则化参数
        
    solver : str, 默认='lasso'
        稀疏回归求解器，可选项: 'lasso', 'ridge', 'elastic_net'
        
    normalize : bool, 默认=True
        是否标准化特征
        
    threshold : float, 默认=0.01
        系数阈值，绝对值小于此值的系数将被设为0
        
    n_jobs : int, 默认=1
        并行作业数
        
    random_state : int, 默认=42
        随机种子
    """
    def __init__(self,
                 poly_degree: int = 3,
                 include_trig: bool = False,
                 include_exp: bool = False,
                 feature_library: Optional[List[str]] = None,
                 alpha: float = 0.1,
                 solver: str = 'lasso',
                 normalize: bool = True,
                 threshold: float = 0.01,
                 n_jobs: int = 1,
                 random_state: int = RANDOM_STATE):
        
        self.poly_degree = poly_degree
        self.include_trig = include_trig
        self.include_exp = include_exp
        self.feature_library = feature_library
        self.alpha = alpha
        self.solver = solver
        self.normalize = normalize
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # 内部状态
        self._model = None
        self._feature_names = None
        self._scaler = None
        self._terms = None  # 存储每个特征的名称
        self._fitted = False
    
    def _build_feature_names(self, X):
        """
        根据原始特征名构建扩展特征名
        """
        input_features = X.columns.tolist()
        feature_names = []
        
        # 常数项
        feature_names.append('1')
        
        # 一次项
        feature_names.extend(input_features)
        
        # 多项式项 (二次及更高)
        poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        poly.fit(np.zeros((1, len(input_features))))
        
        # 获取多项式特征名称
        powers = poly.powers_[len(input_features):]  # 跳过一次项
        
        for power in powers:
            name = ""
            for i, p in enumerate(power):
                if p > 0:
                    if name != "":
                        name += "*"
                    feature = input_features[i]
                    if p == 1:
                        name += feature
                    else:
                        name += f"{feature}^{p}"
            feature_names.append(name)
        
        # 三角函数特征
        if self.include_trig:
            for f in input_features:
                feature_names.append(f"sin({f})")
                feature_names.append(f"cos({f})")
        
        # 指数和对数特征
        if self.include_exp:
            for f in input_features:
                feature_names.append(f"exp({f})")
                feature_names.append(f"log(|{f}|)")
        
        return feature_names
    
    def _generate_features(self, X):
        """
        根据设置生成特征
        """
        X_numpy = X.values
        n_samples, n_features = X_numpy.shape
        
        # 存储所有生成的特征
        feature_list = []
        
        # 常数项
        feature_list.append(np.ones((n_samples, 1)))
        
        # 一次项
        feature_list.append(X_numpy)
        
        # 多项式项
        if self.poly_degree > 1:
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            poly_features = poly.fit_transform(X_numpy)
            # 移除一次项，因为已经添加过了
            poly_features = poly_features[:, n_features:]
            feature_list.append(poly_features)
        
        # 三角函数特征
        if self.include_trig:
            sin_features = np.sin(X_numpy)
            cos_features = np.cos(X_numpy)
            feature_list.append(sin_features)
            feature_list.append(cos_features)
        
        # 指数和对数特征
        if self.include_exp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp_features = np.exp(X_numpy)
                # 防止对0或负数取对数
                log_features = np.log(np.abs(X_numpy) + 1e-10)
            feature_list.append(exp_features)
            feature_list.append(log_features)
        
        # 合并所有特征
        all_features = np.hstack(feature_list)
        
        return all_features
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        y: Union[np.ndarray, pd.Series, list],
    ):
        """拟合SINDy模型"""
        logger.info("开始SINDy模型训练...")

        X, y = auto_convert_input(X, y)

        # 记录特征名称
        self._feature_names = X.columns.tolist()

        # 生成扩展特征名
        self._terms = self._build_feature_names(X)

        # 生成特征
        X_features = self._generate_features(X)
        
        # 创建标准化器
        if self.normalize:
            self._scaler = StandardScaler()
            X_features = self._scaler.fit_transform(X_features)
        
        # 选择稀疏回归模型
        if self.solver == 'lasso':
            model = Lasso(alpha=self.alpha, random_state=self.random_state)
        elif self.solver == 'ridge':
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.solver == 'elastic_net':
            model = ElasticNet(alpha=self.alpha, l1_ratio=0.5, random_state=self.random_state)
        else:
            raise ValueError(f"未知的求解器: {self.solver}")
        
        # 训练模型
        self._model = model.fit(X_features, y)
        
        # 截断小系数
        self._model.coef_ = np.where(np.abs(self._model.coef_) < self.threshold, 0, self._model.coef_)
        
        # 计算模型性能
        y_pred = self._model.predict(X_features)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"SINDy模型训练完成，MSE: {mse:.6f}, R^2: {r2:.6f}")
        logger.info(f"非零系数数量: {np.sum(self._model.coef_ != 0)}/{len(self._model.coef_)}")

        self._fitted = True
        self._train_X = X
        self._train_y = y
        return self
    
    def predict(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        """使用拟合好的模型进行预测"""
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        X = ensure_pandas_dataframe(X, feature_names=self._feature_names)

        # 生成特征
        X_features = self._generate_features(X)

        # 标准化
        if self.normalize:
            X_features = self._scaler.transform(X_features)

        # 预测
        return self._model.predict(X_features)
    
    def get_equation(self) -> str:
        """
        获取发现的方程形式
        
        返回:
        -----
        str
            方程字符串表示
        """
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        intercept = self._model.intercept_
        coefs = self._model.coef_
        
        # 构建方程字符串
        equation = []
        
        # 添加截距项
        if abs(intercept) > self.threshold:
            equation.append(f"{intercept:.4f}")
        
        # 添加其他项
        for i, coef in enumerate(coefs):
            if abs(coef) > self.threshold:
                sign = "+" if coef > 0 else "-"
                if i > 0 and equation:  # 不是第一项且已有项
                    equation.append(f" {sign} ")
                elif i == 0 and not equation and coef < 0:  # 第一项是负数且没有截距
                    equation.append("-")
                elif i > 0 and not equation and coef > 0:  # 不是第一项但没有前面的项，并且是正数
                    pass  # 不需要添加符号
                elif i > 0 and not equation and coef < 0:  # 不是第一项但没有前面的项，并且是负数
                    equation.append("-")
                
                # 添加系数和特征
                term = self._terms[i]
                if term == '1':  # 常数项
                    equation.append(f"{abs(coef):.4f}")
                else:
                    equation.append(f"{abs(coef):.4f}*{term}")
        
        return "".join(equation)
    
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
        
        # 获取非零系数
        coefs = self._model.coef_
        nonzero_indices = np.where(np.abs(coefs) > self.threshold)[0]
        
        nonzero_terms = {}
        for idx in nonzero_indices:
            term = self._terms[idx]
            coef = coefs[idx]
            nonzero_terms[term] = coef
        
        return {
            "equation": self.get_equation(),
            "intercept": self._model.intercept_,
            "nonzero_terms": nonzero_terms,
            "alpha": self.alpha,
            "solver": self.solver,
            "threshold": self.threshold,
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
            "title": "SINDy 回归分析报告",
            "configuration": {
                "poly_degree": self.poly_degree,
                "solver": self.solver,
                "threshold": self.threshold
            },
            "results": {
                "final_model": {
                    "formula_latex": self.get_equation(),
                    "formula_sympy": self.get_equation(),
                    "intercept": self._model.intercept_,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": int(np.sum(self._model.coef_ != 0))
            }
        }

        return SissoReport(report)

    @property
    def explain(self):
        return self._build_report()
