# -*- coding: utf-8 -*-
"""
Hybrid methods for symbolic regression.
"""

from .evolutionary_gradient import EvolutionaryGradientHybrid
from .physics_informed import PhysicsInformedSymbolicRegression
from .multi_objective import MultiObjectiveSymbolicRegression

__all__ = [
    'EvolutionaryGradientHybrid',
    'PhysicsInformedSymbolicRegression', 
    'MultiObjectiveSymbolicRegression'
]
