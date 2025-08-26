# -*- coding: utf-8 -*-
"""
Basic algebraic operators.
"""
import numpy as np
from .base import Operator, register_operator
from ..config import EPSILON
from ..dsl.dimension import Dimension

# --- Dimensional Transforms for Algebra ---

def _add_sub_transform(d1: Dimension, d2: Dimension) -> Dimension:
    if d1 != d2:
        raise TypeError(f"Cannot add/subtract quantities with different dimensions: {d1.to_string()} and {d2.to_string()}")
    return d1

def _mul_transform(d1: Dimension, d2: Dimension) -> Dimension:
    return d1 + d2

def _div_transform(d1: Dimension, d2: Dimension) -> Dimension:
    return d1 - d2

# Addition
class Add(Operator):
    def __init__(self):
        super().__init__(name='+', arity=2, complexity_cost=1, is_binary=True, 
                         latex_fmt='{0} + {1}',
                         dimensional_transform=_add_sub_transform)
    
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

# Subtraction
class Sub(Operator):
    def __init__(self):
        super().__init__(name='-', arity=2, complexity_cost=1, is_binary=True, 
                         latex_fmt='{0} - {1}',
                         dimensional_transform=_add_sub_transform)
        
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

# Multiplication
class Mul(Operator):
    def __init__(self):
        super().__init__(name='*', arity=2, complexity_cost=1, is_binary=True, 
                         latex_fmt='{0} \\times {1}',
                         dimensional_transform=_mul_transform)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

# Safe Division
def _safe_div_validity_checker(a, b):
    return np.abs(b) > EPSILON

class SafeDiv(Operator):
    def __init__(self):
        super().__init__(name='safe_div', arity=2, complexity_cost=2, is_binary=True, 
                         latex_fmt='\\frac{{{0}}}{{{1}}}',
                         validity_checker=_safe_div_validity_checker,
                         dimensional_transform=_div_transform)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Note: The validity check should ideally prevent this from being called
        # with b close to zero, but we add a safe division here as a fallback.
        return np.divide(a, b, out=np.full_like(a, np.nan), where=np.abs(b) > EPSILON)

# Register all operators
register_operator(Add())
register_operator(Sub())
register_operator(Mul())
register_operator(SafeDiv())
