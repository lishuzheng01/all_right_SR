# -*- coding: utf-8 -*-
"""
Base classes for symbolic operators.
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional, Union, List
import numpy as np

from ..dsl.dimension import Dimension, DIMENSIONLESS

# Global registry for operators
_OPERATOR_REGISTRY: Dict[str, 'Operator'] = {}

class Operator(ABC):
    """
    Abstract base class for a mathematical operator used in SISSO.
    """
    def __init__(self,
                 name: str,
                 arity: int,
                 complexity_cost: int = 1,
                 is_binary: bool = False,
                 latex_fmt: Optional[str] = None,
                 validity_checker: Optional[Callable[..., np.ndarray]] = None,
                 dimensional_transform: Optional[Callable[..., Dimension]] = None):
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string.")
        if not isinstance(arity, int) or arity < 1:
            raise ValueError("Operator arity must be a positive integer.")
        
        self.name = name
        self.arity = arity
        self.complexity_cost = complexity_cost
        self.is_binary = is_binary
        self._latex_fmt = latex_fmt
        self.validity_checker = validity_checker
        self.dimensional_transform = dimensional_transform

    @abstractmethod
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        """Executes the operator on the given arguments."""
        raise NotImplementedError

    def is_valid(self, *args: np.ndarray) -> np.ndarray:
        """
        Checks if the operator can be applied to the given arguments.
        Returns a boolean array indicating validity for each element.
        """
        if self.validity_checker:
            return self.validity_checker(*args)
        return np.ones(args[0].shape, dtype=bool)

    def get_output_dimension(self, *input_dims: Dimension) -> Dimension:
        """
        Calculates the output dimension based on input dimensions.
        Returns the output dimension or raises an error if the combination is invalid.
        """
        if len(input_dims) != self.arity:
            raise ValueError(f"Operator '{self.name}' expects {self.arity} dimensions, but got {len(input_dims)}.")

        if self.dimensional_transform:
            return self.dimensional_transform(*input_dims)
        
        # Default behavior for simple functions if no transform is provided
        if self.arity == 1:
            # Assume dimensionless output for functions like log, sin, etc.
            # unless specified otherwise. This is a heuristic.
            return DIMENSIONLESS
        
        raise NotImplementedError(f"Dimensional transform not implemented for operator '{self.name}'.")


    @property
    def latex_fmt(self) -> str:
        """LaTeX format string for the operator."""
        if self._latex_fmt:
            return self._latex_fmt
        if self.arity == 1:
            return f"\\mathrm{{{self.name}}}({{}})"
        elif self.arity == 2 and self.is_binary:
            # For binary operators like +, -, *, /
            return f"{{}} {self.name} {{}}"
        else:
            # For function-like operators like pow(a, b)
            args = ", ".join(["{}"] * self.arity)
            return f"\\mathrm{{{self.name}}}({args})"

    def __repr__(self) -> str:
        return f"Operator(name='{self.name}', arity={self.arity})"

def register_operator(op_instance: Operator):
    """Registers an operator instance in the global registry."""
    if not isinstance(op_instance, Operator):
        raise TypeError("Can only register instances of the Operator class.")
    if op_instance.name in _OPERATOR_REGISTRY:
        # For simplicity in this project, we can allow overwriting,
        # but in a real library, a warning or error might be better.
        pass
    _OPERATOR_REGISTRY[op_instance.name] = op_instance

def get_operator(name: str) -> Operator:
    """Retrieves an operator from the registry by its name."""
    if name not in _OPERATOR_REGISTRY:
        raise KeyError(f"Operator '{name}' not found in registry. "
                       f"Available operators: {list(_OPERATOR_REGISTRY.keys())}")
    return _OPERATOR_REGISTRY[name]

def get_all_operators() -> Dict[str, Operator]:
    """Returns a copy of the operator registry."""
    return _OPERATOR_REGISTRY.copy()

def clear_registry():
    """Clears the operator registry. Mainly for testing."""
    _OPERATOR_REGISTRY.clear()
