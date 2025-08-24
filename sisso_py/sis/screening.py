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
    """Scores features based on the absolute Pearson correlation with the target."""
    if X.empty:
        return []
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    return list(correlations.head(top_k).index)

def mutual_information_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """Scores features based on mutual information with the target."""
    if X.empty:
        return []
    mi_scores = mutual_info_regression(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return list(mi_series.head(top_k).index)

def random_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """Randomly selects features."""
    if X.empty:
        return []
    np.random.seed(42)  # 为了可重复性
    available_features = list(X.columns)
    n_select = min(top_k, len(available_features))
    return list(np.random.choice(available_features, size=n_select, replace=False))

def variance_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """Scores features based on their variance."""
    if X.empty:
        return []
    variances = X.var().sort_values(ascending=False)
    return list(variances.head(top_k).index)

def f_regression_screener(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    """Scores features based on F-statistic from univariate linear regression."""
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
    """Uses Recursive Feature Elimination with Linear Regression."""
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
    """Uses LASSO regularization path to select features."""
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
    """Combines multiple screening methods using voting."""
    if X.empty:
        return []
    
    try:
        # 获取各种方法的结果
        methods = [
            ('pearson', pearson_correlation_screener),
            ('mutual_info', mutual_information_screener),
            ('f_regression', f_regression_screener),
            ('variance', variance_screener)
        ]
        
        # 每种方法选择top_k特征
        all_selections = {}
        for name, method in methods:
            try:
                selected = method(X, y, top_k)
                all_selections[name] = set(selected)
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
    'pearson': pearson_correlation_screener,
    'mutual_info': mutual_information_screener,
    'random': random_screener,
    'variance': variance_screener,
    'f_regression': f_regression_screener,
    'rfe': rfe_screener,
    'lasso_path': lasso_path_screener,
    'combined': combined_screener,
    # 'hsic': hsic_screener # Placeholder for future HSIC implementation
}

class SIS:
    """
    Performs Sure Independence Screening to reduce the feature space.
    """
    def __init__(self,
                 screener: str = 'pearson',
                 top_k: int = 1000):
        if screener not in SCREENERS:
            raise ValueError(f"Unknown screener '{screener}'. "
                             f"Available options: {list(SCREENERS.keys())}")
        self.screener = SCREENERS[screener]
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
