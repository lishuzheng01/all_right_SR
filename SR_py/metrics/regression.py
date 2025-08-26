# -*- coding- utf-8 -*-
"""
Common regression metrics.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates a standard set of regression metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        A dictionary containing RMSE, MAE, and R-squared.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays y_true and y_pred must have the same shape.")

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics
