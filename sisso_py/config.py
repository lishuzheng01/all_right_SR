# -*- coding: utf-8 -*-
"""
Global configurations, default settings, and constants.
"""
import numpy as np

# Default operator set
DEFAULT_OPERATORS = [
    '+', '-', '*', 'safe_div',
    'sqrt', 'cbrt', 'square',
    'log', 'exp',
    'abs', 'sign',
    'sin', 'cos',
]

# Numerical stability constants
EPSILON = 1e-8
CLIP_MIN = -1e10
CLIP_MAX = 1e10

# Default random state for reproducibility
RANDOM_STATE = 42
