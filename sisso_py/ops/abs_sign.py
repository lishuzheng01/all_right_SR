# -*- coding: utf-8 -*-
"""
Absolute value and sign operators.
"""
import numpy as np
from .base import Operator, register_operator
from ..dsl.dimension import Dimension

# --- Dimensional Transforms ---
def _identity_transform(d: Dimension) -> Dimension:
    """The dimension is unchanged."""
    return d

class Abs(Operator):
    def __init__(self):
        super().__init__(name='abs', arity=1, complexity_cost=1, 
                         latex_fmt='|{0}|',
                         dimensional_transform=_identity_transform)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

class Sign(Operator):
    def __init__(self):
        super().__init__(name='sign', arity=1, complexity_cost=2, 
                         latex_fmt='\\mathrm{{sign}}({0})',
                         dimensional_transform=_identity_transform)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Rule: sign(0)=0
        return np.sign(x)

# --- Registration ---
register_operator(Abs())
register_operator(Sign())
