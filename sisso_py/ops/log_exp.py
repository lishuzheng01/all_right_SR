# -*- coding: utf-8 -*-
"""
Logarithmic and exponential operators.
"""
import numpy as np
from .base import Operator, register_operator
from ..config import EPSILON
from ..dsl.dimension import Dimension, DIMENSIONLESS

# --- Validity Checkers ---
def _safe_log_validity_checker(x):
    return x > EPSILON

# --- Dimensional Transforms ---
def _transcendental_transform(d: Dimension) -> Dimension:
    """Input must be dimensionless, output is dimensionless."""
    if not d.is_dimensionless():
        raise TypeError(f"Input to transcendental function must be dimensionless, but got {d.to_string()}")
    return DIMENSIONLESS

# --- Operators ---
class SafeLog(Operator):
    def __init__(self):
        super().__init__(name='log', arity=1, complexity_cost=2,
                         latex_fmt='\\log({0})',
                         validity_checker=_safe_log_validity_checker,
                         dimensional_transform=_transcendental_transform)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Based on the rule: log(safe_abs(x)+eps)
        safe_x = np.abs(x) + EPSILON
        return np.log(safe_x)

class SafeLog10(Operator):
    def __init__(self):
        super().__init__(name='log10', arity=1, complexity_cost=2,
                         latex_fmt='\\log_{{10}}({0})',
                         validity_checker=_safe_log_validity_checker,
                         dimensional_transform=_transcendental_transform)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        safe_x = np.abs(x) + EPSILON
        return np.log10(safe_x)
        
class Exp(Operator):
    def __init__(self):
        super().__init__(name='exp', arity=1, complexity_cost=3, 
                         latex_fmt='e^{{{0}}}',
                         dimensional_transform=_transcendental_transform)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # np.exp handles overflow gracefully by returning inf, which will be clipped later.
        return np.exp(x)

# --- Registration ---
register_operator(SafeLog())
register_operator(SafeLog10())
register_operator(Exp())
