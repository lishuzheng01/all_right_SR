# -*- coding: utf-8 -*-
"""
SISSO-Py: A Python implementation of the SISSO method.
"""

__version__ = "0.1.0"

from .model import SissoRegressor
from .interface import create_regressor
from .utils.logging import setup_logging

# Setup default logging
setup_logging()
