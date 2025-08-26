# -*- coding: utf-8 -*-
"""
Complexity management and K-level generation control.
"""

class ComplexityBudget:
    """
    A simple class to manage the complexity budget for feature generation.
    """
    def __init__(self, max_total_complexity: int):
        if not isinstance(max_total_complexity, int) or max_total_complexity < 1:
            raise ValueError("Max total complexity must be a positive integer.")
        self.max_total_complexity = max_total_complexity

    def is_within_budget(self, expr_complexity: int) -> bool:
        """
        Checks if a given expression complexity is within the budget.
        """
        return expr_complexity <= self.max_total_complexity

    def __repr__(self) -> str:
        return f"ComplexityBudget(max_total_complexity={self.max_total_complexity})"
