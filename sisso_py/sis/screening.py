# -*- coding: utf-8 -*-
"""
Sure Independence Screening (SIS) methods.
"""
from typing import List, Dict, Callable, Optional
import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import (
    mutual_info_regression, 
    f_regression, 
    VarianceThreshold,
    RFE
)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def pearson_correlation_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Scores features based on the absolute Pearson correlation with the target.
    
    适用于探索阶段：快速了解数据特征之间的线性关系强度。
    Pearson相关系数衡量了特征与目标变量之间的线性相关性，
    范围从-1（完全负相关）到1（完全正相关）。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        按相关性排序的顶部特征名称列表
    """
    if X.empty:
        return []
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    return list(correlations.head(top_k).index)

def mutual_information_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Scores features based on mutual information with the target.
    
    适用于非线性关系：能捕获复杂的非线性模式和依赖关系。
    互信息测量特征与目标变量之间的统计依赖性，不限于线性关系，
    能够发现Pearson相关系数可能会忽略的关系模式。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        按互信息分数排序的顶部特征名称列表
    """
    if X.empty:
        return []
    mi_scores = mutual_info_regression(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return list(mi_series.head(top_k).index)

def random_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Randomly selects features.
    
    适用于基线对比：提供随机选择的特征作为基准比较。
    随机选择特征是评估其他特征选择方法有效性的重要基线，
    如果其他方法的性能与随机选择相近，可能表明它们效果不佳。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        随机选择的特征名称列表
    """
    if X.empty:
        return []
    np.random.seed(42)  # 为了可重复性
    available_features = list(X.columns)
    n_select = min(top_k, len(available_features))
    return list(np.random.choice(available_features, size=n_select, replace=False))

def variance_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Scores features based on their variance.
    
    适用于预处理步骤：移除低变异性特征。
    方差筛选是最基本的无监督特征选择方法，
    它假设方差接近零的特征几乎没有信息量。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        按方差排序的顶部特征名称列表
    """
    if X.empty:
        return []
    variances = X.var().sort_values(ascending=False)
    return list(variances.head(top_k).index)

def f_regression_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Scores features based on F-statistic from univariate linear regression.
    
    适用于线性关系：专门针对线性回归模型的特征选择。
    F统计量衡量每个特征单独用于预测目标变量的能力，
    通过计算每个特征的线性回归模型与零模型之间的F检验。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        按F统计量排序的顶部特征名称列表
    """
    if X.empty:
        return []
    try:
        f_scores, _ = f_regression(X, y)
        f_series = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
        return list(f_series.head(top_k).index)
    except Exception as e:
        logger.warning(f"F-regression screening failed: {e}. Falling back to variance screening.")
        return variance_screener(X, y, top_k)

def rfe_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Uses Recursive Feature Elimination with Linear Regression.
    
    适用于精确建模：通过反复训练模型并移除最不重要的特征来精确选择特征。
    RFE通过迭代训练模型，每次移除最不重要的特征，直到达到指定数量的特征，
    这种方法考虑了特征之间的相互作用，而不仅仅是单变量关系。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        通过递归特征消除选择的特征名称列表
    """
    if X.empty:
        return []
    try:
        # 数据标准化，避免数值问题
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # 使用线性回归作为基础估计器
        estimator = LinearRegression()
        n_select = min(top_k, X.shape[1])
        
        rfe = RFE(estimator=estimator, n_features_to_select=n_select)
        rfe.fit(X_scaled_df, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        return selected_features
    except Exception as e:
        logger.warning(f"RFE screening failed: {e}. Falling back to correlation screening.")
        return pearson_correlation_screener(X, y, top_k)

def lasso_path_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Uses LASSO regularization path to select features.
    
    适用于高维数据：处理特征维度灾难和高度相关特征。
    通过在不同正则化强度下运行LASSO，观察哪些特征在大多数情况下被选中，
    这种方法能够在存在共线性时有效识别重要特征，并处理大量特征。
    
    Args:
        X: 特征矩阵
        y: 目标变量
        top_k: 要选择的顶部特征数量
        
    Returns:
        按LASSO路径中出现频率排序的顶部特征名称列表
    """
    if X.empty:
        return []
    try:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 使用较强的正则化，然后逐步降低
        alphas = np.logspace(-4, 0, 50)
        lasso = Lasso(max_iter=1000)
        
        # 记录每个特征被选中的频率
        feature_counts = np.zeros(X.shape[1])
        
        for alpha in alphas:
            lasso.set_params(alpha=alpha)
            lasso.fit(X_scaled, y)
            feature_counts += (np.abs(lasso.coef_) > 1e-6)
        
        # 按选中频率排序
        frequency_series = pd.Series(feature_counts, index=X.columns).sort_values(ascending=False)
        return list(frequency_series.head(top_k).index)
    except Exception as e:
        logger.warning(f"LASSO path screening failed: {e}. Falling back to correlation screening.")
        return pearson_correlation_screener(X, y, top_k)

def combined_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """
    Combines multiple screening methods using voting.
    
    这是一个稳健的筛选方法，通过综合多种筛选算法的结果，
    对特征进行投票排序，能够从多角度验证特征的重要性。
    """
    if X.empty:
        return []
    
    try:
        # 获取各种方法的结果（除了'random'和'combined'自身）
        methods = [
            ('pearson', pearson_correlation_screener),
            ('mutual_info', mutual_information_screener),
            ('f_regression', f_regression_screener),
            ('variance', variance_screener),
            ('lasso_path', lasso_path_screener),
            ('rfe', rfe_screener)
        ]
        
        # 每种方法选择top_k特征
        all_selections = {}
        for name, method in methods:
            try:
                selected = method(X, y, top_k)
                all_selections[name] = set(selected)
                logger.info(f"Combined screener: {name} selected {len(selected)} features")
            except Exception as e:
                logger.warning(f"Method {name} failed: {e}")
                continue
        
        if not all_selections:
            return pearson_correlation_screener(X, y, top_k)
        
        # 统计每个特征被选中的次数
        feature_votes = {}
        for feature in X.columns:
            votes = sum(1 for selection in all_selections.values() if feature in selection)
            feature_votes[feature] = votes
        
        # 按投票数排序，然后按相关性排序（打破平局）
        correlations = X.corrwith(y).abs()
        sorted_features = sorted(
            feature_votes.keys(),
            key=lambda x: (feature_votes[x], correlations[x]),
            reverse=True
        )
        
        return sorted_features[:top_k]
    except Exception as e:
        logger.warning(f"Combined screening failed: {e}. Falling back to correlation screening.")
        return pearson_correlation_screener(X, y, top_k)

# A registry for different screening methods
SCREENERS: Dict[str, Callable] = {
    'pearson': pearson_correlation_screener,        # 探索阶段：快速了解数据特征
    'mutual_info': mutual_information_screener,     # 非线性关系：能捕获复杂模式
    'random': random_screener,                      # 基线对比：随机基线
    'variance': variance_screener,                  # 方差筛选：移除低变异性特征
    'f_regression': f_regression_screener,          # 线性关系：专门针对线性关系
    'rfe': rfe_screener,                            # 精确建模：递归特征消除
    'lasso_path': lasso_path_screener,              # 高维数据：处理特征维度灾难
    'combined': combined_screener,                  # 稳健建模：多角度验证
    # 'hsic': hsic_screener                         # Placeholder for future HSIC implementation
}

class SIS:
    """
    Performs Sure Independence Screening to reduce the feature space.
    
    SIS (Sure Independence Screening) 是一种用于高维数据的特征选择方法，
    通过快速筛选减少特征空间，保留最相关的特征。该类提供了多种筛选算法选择，
    可以根据不同的数据特性和建模目标选择合适的方法。
    
    可用筛选方法:
    - 'pearson': 探索阶段，快速了解数据特征之间的线性关系
    - 'f_regression': 线性关系，专门针对线性回归模型的特征选择
    - 'mutual_info': 非线性关系，能捕获复杂的非线性模式和依赖关系
    - 'lasso_path': 高维数据，处理特征维度灾难和高度相关特征
    - 'combined': 稳健建模，通过多种方法投票选择特征，多角度验证
    - 'rfe': 精确建模，通过递归特征消除进行精确选择
    - 'random': 基线对比，提供随机基线用于评估其他方法
    - 'variance': 预处理步骤，移除低变异性特征
    """
    def __init__(self,
                 screener: str = 'pearson',
                 top_k: int = 1000):
        """
        初始化SIS实例
        
        Args:
            screener: 要使用的筛选方法名称，必须在SCREENERS字典中注册
            top_k: 要选择的顶部特征数量
        """
        if screener not in SCREENERS:
            raise ValueError(f"Unknown screener '{screener}'. "
                             f"Available options: {list(SCREENERS.keys())}")
        self.screener = SCREENERS[screener]
        self.screener_name = screener  # 保存筛选器名称，便于后续记录和报告
        self.top_k = top_k
        logger.info(f"SIS initialized with screener='{screener}' and top_k={top_k}.")

    def screen(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply the screening method to the feature matrix X.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.

        Returns:
            pd.DataFrame: A DataFrame containing only the top-k features.
        """
        if X.shape[1] <= self.top_k:
            logger.info("Number of features is less than or equal to top_k. Skipping screening.")
            return X

        logger.info(f"Screening {X.shape[1]} features to select top {self.top_k}...")
        
        selected_features = self.screener(X, y, self.top_k)
        
        logger.info(f"Selected {len(selected_features)} features after screening.")
        
        return X[selected_features]
