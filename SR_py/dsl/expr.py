# -*- coding: utf-8 -*-
"""
Core data structures for representing symbolic expressions as trees.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import numpy as np
import sympy

from ..ops.base import Operator, get_operator
from .dimension import Dimension, DIMENSIONLESS

class Expr(ABC):
    """Abstract base class for an expression node."""
    
    @property
    @abstractmethod
    def dimension(self) -> Dimension:
        """The physical dimension of the expression."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the expression tree given a data dictionary."""
        raise NotImplementedError

    @abstractmethod
    def to_sympy(self) -> sympy.Expr:
        """Convert the expression to a SymPy expression."""
        raise NotImplementedError

    @abstractmethod
    def to_latex(self) -> str:
        """Convert the expression to a LaTeX string."""
        raise NotImplementedError
    
    @abstractmethod
    def get_signature(self) -> str:
        """Get a canonical string representation (signature) for the expression."""
        raise NotImplementedError

    @abstractmethod
    def get_complexity(self) -> int:
        """Calculate the complexity of the expression."""
        raise NotImplementedError
        
    def __hash__(self):
        return hash(self.get_signature())

    def __eq__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return self.get_signature() == other.get_signature()
        
    def __repr__(self):
        return self.get_signature()

class Var(Expr):
    """Represents a variable."""
    def __init__(self, name: str, dimension: Dimension = DIMENSIONLESS):
        if not isinstance(name, str) or not name:
            raise ValueError("Variable name must be a non-empty string.")
        self.name = name
        self.complexity = 1
        self._dimension = dimension

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.name not in data:
            raise KeyError(f"Variable '{self.name}' not found in the provided data.")
        return data[self.name]

    def to_sympy(self) -> sympy.Expr:
        return sympy.Symbol(self.name)

    def to_latex(self) -> str:
        return self.name.replace('_', '\\_')

    def get_signature(self) -> str:
        return self.name
        
    def get_complexity(self) -> int:
        return self.complexity

class Const(Expr):
    """Represents a constant value."""
    def __init__(self, value: float):
        self.value = float(value)
        self.complexity = 1
        self._dimension = DIMENSIONLESS
    
    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        # Get a shape from the data to broadcast the constant
        some_key = next(iter(data))
        return np.full_like(data[some_key], self.value)

    def to_sympy(self) -> sympy.Expr:
        return sympy.Number(self.value)

    def to_latex(self) -> str:
        return f"{self.value:.4g}"

    def get_signature(self) -> str:
        return str(self.value)
        
    def get_complexity(self) -> int:
        return self.complexity

class Unary(Expr):
    """Represents a unary operation."""
    def __init__(self, op_name: str, child: Expr):
        self.op: Operator = get_operator(op_name)
        if self.op.arity != 1:
            raise ValueError(f"Operator '{op_name}' is not unary (arity={self.op.arity}).")
        self.child = child
        self.complexity = self.op.complexity_cost + self.child.get_complexity()
        self._dimension = self.op.get_output_dimension(self.child.dimension)
        
    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        child_val = self.child.evaluate(data)
        return self.op(child_val)

    def to_sympy(self) -> sympy.Expr:
        child_sympy = self.child.to_sympy()
        # This requires a mapping from our op names to sympy functions
        # For now, we'll create an undefined function.
        sympy_func = sympy.Function(self.op.name)
        return sympy_func(child_sympy)
        
    def to_latex(self) -> str:
        child_latex = self.child.to_latex()
        return self.op.latex_fmt.format(f"({child_latex})")

    def get_signature(self) -> str:
        return f"{self.op.name}({self.child.get_signature()})"
        
    def get_complexity(self) -> int:
        return self.complexity

class Binary(Expr):
    """Represents a binary operation."""
    def __init__(self, op_name: str, left: Expr, right: Expr):
        self.op: Operator = get_operator(op_name)
        if self.op.arity != 2:
            raise ValueError(f"Operator '{op_name}' is not binary (arity={self.op.arity}).")
        
        # For commutative operators like '+' and '*', enforce an order to ensure
        # a canonical signature (e.g., a+b and b+a have the same signature).
        if self.op.name in ('+', '*') and left.get_signature() > right.get_signature():
            self.left, self.right = right, left
        else:
            self.left = left
            self.right = right
            
        self.complexity = self.op.complexity_cost + self.left.get_complexity() + self.right.get_complexity()
        self._dimension = self.op.get_output_dimension(self.left.dimension, self.right.dimension)
            
    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        left_val = self.left.evaluate(data)
        right_val = self.right.evaluate(data)
        return self.op(left_val, right_val)

    def to_sympy(self) -> sympy.Expr:
        left_sympy = self.left.to_sympy()
        right_sympy = self.right.to_sympy()
        # This requires a mapping as well.
        if self.op.name == '+': return left_sympy + right_sympy
        if self.op.name == '-': return left_sympy - right_sympy
        if self.op.name == '*': return left_sympy * right_sympy
        if self.op.name == 'safe_div': return left_sympy / right_sympy
        
        sympy_func = sympy.Function(self.op.name)
        return sympy_func(left_sympy, right_sympy)

    def to_latex(self) -> str:
        left_latex = self.left.to_latex()
        right_latex = self.right.to_latex()
        
        # 智能括号管理：避免不必要的括号
        left_needs_parens = self._needs_parentheses(self.left)
        right_needs_parens = self._needs_parentheses(self.right)
        
        left_formatted = f"({left_latex})" if left_needs_parens else left_latex
        right_formatted = f"({right_latex})" if right_needs_parens else right_latex
        
        return self.op.latex_fmt.format(left_formatted, right_formatted)
    
    def _needs_parentheses(self, expr) -> bool:
        """判断表达式是否需要括号"""
        # 变量和常数不需要括号
        if isinstance(expr, (Var, Const)):
            return False
            
        # 对于二元运算，根据运算符优先级判断
        if isinstance(expr, Binary):
            # 加减法在乘除法中需要括号
            if self.op.name in ('*', 'safe_div') and expr.op.name in ('+', '-'):
                return True
            # 除法的分母如果是运算表达式需要括号
            if self.op.name == 'safe_div' and expr == self.right and expr.op.name in ('+', '-', '*', 'safe_div'):
                return True
        
        return False

    def get_signature(self) -> str:
        return f"{self.op.name}({self.left.get_signature()},{self.right.get_signature()})"
        
    def get_complexity(self) -> int:
        return self.complexity
