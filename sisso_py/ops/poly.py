# -*- coding: utf-8 -*-
"""
Polynomial expansion operators.
"""
import numpy as np
from .base import Operator, register_operator
from ..dsl.dimension import Dimension

class Polynomial(Operator):
    """
    Represents a polynomial term x^n.
    This is functionally similar to the Pow operator but is categorized
    here for clarity and potential future extensions like Chebyshev polynomials.
    """
    def __init__(self, degree: int):
        if not isinstance(degree, int) or degree < 2:
            raise ValueError("Polynomial degree must be an integer >= 2.")
        
        self.degree = degree
        name = f"poly{degree}"
        # Use a more explicit LaTeX format
        latex_fmt = f"({{0}})^{{{degree}}}"
        
        # Complexity can be scaled with degree
        complexity_cost = degree -1

        super().__init__(name=name,
                         arity=1,
                         complexity_cost=complexity_cost,
                         latex_fmt=latex_fmt,
                         dimensional_transform=lambda d: d * self.degree)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.power(x, self.degree)

# --- Registration of common polynomial degrees ---
# The user can register more degrees if they wish.
# Note: degree 2 is identical to the 'square' operator. We register it
# under a different name for users who might explicitly look for polynomials.
register_operator(Polynomial(2))
register_operator(Polynomial(3))
