# -*- coding: utf-8 -*-
"""
Power and root operators.
"""
import numpy as np
from .base import Operator, register_operator
from ..dsl.dimension import Dimension

# --- Validity Checkers ---
def _sqrt_validity_checker(x):
    return x >= 0

def _cbrt_validity_checker(x):
    # cbrt is defined for all real numbers
    return np.ones_like(x, dtype=bool)

# --- Dimensional Transforms ---
def _power_transform(d: Dimension, exponent: float) -> Dimension:
    return d * exponent

# --- Operators ---
class Sqrt(Operator):
    def __init__(self):
        super().__init__(name='sqrt', arity=1, complexity_cost=2, 
                         latex_fmt='\\sqrt{{{0}}}',
                         validity_checker=_sqrt_validity_checker,
                         dimensional_transform=lambda d: _power_transform(d, 0.5))
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x, out=np.full_like(x, np.nan), where=x >= 0)

class Cbrt(Operator):
    def __init__(self):
        super().__init__(name='cbrt', arity=1, complexity_cost=2, 
                         latex_fmt='\\sqrt[3]{{{0}}}',
                         validity_checker=_cbrt_validity_checker,
                         dimensional_transform=lambda d: _power_transform(d, 1/3))
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.cbrt(x)

class Square(Operator):
    def __init__(self):
        super().__init__(name='square', arity=1, complexity_cost=1, 
                         latex_fmt='{0}^{{2}}',
                         dimensional_transform=lambda d: _power_transform(d, 2))
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)

class Pow(Operator):
    def __init__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError(f"Exponent must be a number, but got {type(exponent)}")
        self.exponent = exponent
        name = f"pow{exponent}"
        latex_fmt = f"({{0}})^{{{exponent}}}"
        super().__init__(name=name, arity=1, complexity_cost=abs(exponent), latex_fmt=latex_fmt,
                         dimensional_transform=lambda d: _power_transform(d, self.exponent))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Basic protection for complex numbers, e.g., (-1)^0.5
        if self.exponent < 0 or (0 < self.exponent < 1):
             with np.errstate(invalid='ignore'):
                return np.power(np.abs(x), self.exponent)
        return np.power(x, self.exponent)

# --- Registration ---
register_operator(Sqrt())
register_operator(Cbrt())
register_operator(Square())
# We can register specific powers if needed, e.g.:
# register_operator(Pow(3))
# register_operator(Pow(-1))
