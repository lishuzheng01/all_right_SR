# -*- coding: utf-8 -*-
"""
Common physics-related operators.
(Placeholder for future implementation)
"""
import numpy as np
from .base import Operator, register_operator
from ..dsl.dimension import Dimension, DIMENSIONLESS

# --- Dimensional Transforms ---
def _transcendental_transform(d: Dimension) -> Dimension:
    """Input must be dimensionless, output is dimensionless."""
    if not d.is_dimensionless():
        raise TypeError(f"Input to transcendental function must be dimensionless, but got {d.to_string()}")
    return DIMENSIONLESS

def _reciprocal_transform(d: Dimension) -> Dimension:
    return d * -1.0

# Reciprocal
class Reciprocal(Operator):
    def __init__(self):
        super().__init__(name='reciprocal', arity=1, complexity_cost=2, 
                         latex_fmt='\\frac{{1}}{{{0}}}',
                         dimensional_transform=_reciprocal_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.reciprocal(x.astype(float), out=np.full_like(x, np.nan, dtype=float), where=(np.abs(x) > 1e-8))

# Sine
class Sin(Operator):
    def __init__(self):
        super().__init__(name='sin', arity=1, complexity_cost=3, 
                         latex_fmt='\\sin({0})',
                         dimensional_transform=_transcendental_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

# Cosine
class Cos(Operator):
    def __init__(self):
        super().__init__(name='cos', arity=1, complexity_cost=3, 
                         latex_fmt='\\cos({0})',
                         dimensional_transform=_transcendental_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

# Hyperbolic Sine
class Sinh(Operator):
    def __init__(self):
        super().__init__(name='sinh', arity=1, complexity_cost=3, 
                         latex_fmt='\\sinh({0})',
                         dimensional_transform=_transcendental_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sinh(x)

# Hyperbolic Cosine
class Cosh(Operator):
    def __init__(self):
        super().__init__(name='cosh', arity=1, complexity_cost=3, 
                         latex_fmt='\\cosh({0})',
                         dimensional_transform=_transcendental_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.cosh(x)

# Hyperbolic Tangent
class Tanh(Operator):
    def __init__(self):
        super().__init__(name='tanh', arity=1, complexity_cost=3, 
                         latex_fmt='\\tanh({0})',
                         dimensional_transform=_transcendental_transform)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


register_operator(Reciprocal())
register_operator(Sin())
register_operator(Cos())
register_operator(Sinh())
register_operator(Cosh())
register_operator(Tanh())
