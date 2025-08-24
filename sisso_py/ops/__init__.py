# -*- coding: utf-8 -*-
"""
The ops module contains all the symbolic operators available for feature generation.

By importing the operator modules here, we ensure that all defined operators
are automatically registered in the central registry upon loading the ops package.
"""

from . import algebra
from . import power_root
from . import log_exp
from . import abs_sign
from . import physics
from . import poly

# After the imports above, the registry in .base should be populated.
from .base import get_operator, get_all_operators, Operator, register_operator
