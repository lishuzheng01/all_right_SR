# -*- coding: utf-8 -*-
"""
Data loading and interface utilities.
"""
import pandas as pd
import numpy as np
from typing import Union, Tuple

def load_from_pandas(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates a pandas DataFrame into features (X) and target (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    return X, y

def load_from_numpy(X_np: np.ndarray, y_np: np.ndarray, feature_names: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Converts NumPy arrays into a pandas DataFrame and Series.
    """
    if feature_names is None:
        feature_names = [f'x_{i}' for i in range(X_np.shape[1])]
    
    X = pd.DataFrame(X_np, columns=feature_names)
    y = pd.Series(y_np, name='target')
    
    return X, y
