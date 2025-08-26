# -*- coding: utf-8 -*-
"""
数据类型转换工具
支持numpy数组和pandas对象之间的转换
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
import warnings

def ensure_pandas_dataframe(X: Union[np.ndarray, pd.DataFrame, pd.Series],
                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    确保输入是pandas DataFrame格式
    
    参数:
    -----
    X : np.ndarray, pd.DataFrame or pd.Series
        输入数据。如果为Series，将被转换为单列DataFrame。
    feature_names : List[str], 可选
        特征名称列表
        
    返回:
    -----
    pd.DataFrame
        转换后的DataFrame
    """
    if isinstance(X, pd.DataFrame):
        return X

    # 处理pandas Series
    if isinstance(X, pd.Series):
        col_name = feature_names[0] if feature_names else (X.name or 'x')
        return X.to_frame(name=col_name)

    # 处理numpy数组
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if feature_names is None:
            if X.shape[1] == 1:
                feature_names = ['x']
            else:
                feature_names = [f'x{i}' for i in range(X.shape[1])]

        return pd.DataFrame(X, columns=feature_names)

    raise TypeError(f"不支持的输入类型: {type(X)}")

def ensure_pandas_series(y: Union[np.ndarray, pd.Series, list], 
                        name: str = 'y') -> pd.Series:
    """
    确保输入是pandas Series格式
    
    参数:
    -----
    y : np.ndarray, pd.Series, or list
        目标变量
    name : str
        Series的名称
        
    返回:
    -----
    pd.Series
        转换后的Series
    """
    if isinstance(y, pd.Series):
        return y
    
    if isinstance(y, (np.ndarray, list)):
        return pd.Series(y, name=name)
    
    raise TypeError(f"不支持的目标变量类型: {type(y)}")

def ensure_numpy_array(data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    确保输入是numpy数组格式
    
    参数:
    -----
    data : np.ndarray, pd.DataFrame, or pd.Series
        输入数据
        
    返回:
    -----
    np.ndarray
        转换后的numpy数组
    """
    if isinstance(data, np.ndarray):
        return data
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return np.asarray(data.values)
    
    raise TypeError(f"不支持的数据类型: {type(data)}")

def auto_convert_input(X: Union[np.ndarray, pd.DataFrame, pd.Series],
                      y: Union[np.ndarray, pd.Series, list],
                      feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    自动转换输入数据为pandas格式
    
    参数:
    -----
    X : np.ndarray, pd.DataFrame or pd.Series
        特征数据
    y : np.ndarray, pd.Series, or list
        目标变量
    feature_names : List[str], 可选
        特征名称列表
        
    返回:
    -----
    Tuple[pd.DataFrame, pd.Series]
        转换后的特征和目标变量
    """
    X_df = ensure_pandas_dataframe(X, feature_names)
    y_series = ensure_pandas_series(y)
    
    # 确保索引一致
    if len(X_df) != len(y_series):
        raise ValueError(f"特征数据长度 ({len(X_df)}) 与目标变量长度 ({len(y_series)}) 不匹配")
    
    # 重置索引以确保一致性
    X_df = X_df.reset_index(drop=True)
    y_series = y_series.reset_index(drop=True)
    
    return X_df, y_series

def validate_input_shapes(X: Union[np.ndarray, pd.DataFrame, pd.Series],
                         y: Union[np.ndarray, pd.Series, list]) -> None:
    """
    验证输入数据的形状
    
    参数:
    -----
    X : np.ndarray, pd.DataFrame or pd.Series
        特征数据
    y : np.ndarray, pd.Series, or list
        目标变量
    """
    if isinstance(X, np.ndarray):
        if X.ndim > 2:
            raise ValueError(f"特征数据维度过高: {X.ndim}，最多支持2维")
        if X.ndim == 1:
            warnings.warn("输入的特征数据是1维数组，将自动转换为2维", UserWarning)
    
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            raise ValueError(f"目标变量维度过高: {y.ndim}，只支持1维")
