# -*- coding: utf-8 -*-
"""
Hashing and signature generation for expression deduplication.
"""
from ..dsl.expr import Expr

def get_expr_signature(expr: Expr) -> str:
    """
    Returns the canonical signature of an expression.

    This function just wraps the method on the expression object itself,
    providing a single functional entry point if needed.
    """
    return expr.get_signature()
