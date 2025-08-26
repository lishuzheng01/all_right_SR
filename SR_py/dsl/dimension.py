# -*- coding: utf-8 -*-
"""
(Optional) Dimensional analysis for physical consistency.
"""
import numpy as np
from typing import List, Optional

# A common standard for base dimensions (SI units):
# Mass (M), Length (L), Time (T), Electric Current (I), Temperature (Θ), Amount of Substance (N), Luminous Intensity (J)
BASE_DIMENSIONS = ['M', 'L', 'T', 'I', 'Θ', 'N', 'J']

class Dimension:
    """
    Represents a physical dimension as a vector of exponents of base units.
    
    Example: Velocity is L/T, which corresponds to a dimension vector of
             [0, 1, -1, 0, 0, 0, 0] for (M, L, T, I, Θ, N, J).
    """
    def __init__(self, exponents: List[float]):
        self.vector = np.array(exponents, dtype=float)
        if self.vector.shape != (len(BASE_DIMENSIONS),):
            raise ValueError(f"Dimension vector must have length {len(BASE_DIMENSIONS)}")

    def is_dimensionless(self) -> bool:
        """Checks if the dimension is dimensionless (all exponents are zero)."""
        return np.all(self.vector == 0)

    def __add__(self, other: 'Dimension') -> 'Dimension':
        """Addition of dimensions (for multiplication of quantities)."""
        return Dimension(self.vector + other.vector)

    def __sub__(self, other: 'Dimension') -> 'Dimension':
        """Subtraction of dimensions (for division of quantities)."""
        return Dimension(self.vector - other.vector)
        
    def __mul__(self, scalar: float) -> 'Dimension':
        """Scalar multiplication (for powers of quantities)."""
        return Dimension(self.vector * scalar)
        
    def __rmul__(self, scalar: float) -> 'Dimension':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Dimension':
        """Scalar division (for roots of quantities)."""
        if scalar == 0:
            raise ValueError("Cannot divide dimension by zero.")
        return Dimension(self.vector / scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return NotImplemented
        return np.array_equal(self.vector, other.vector)

    def __repr__(self) -> str:
        return f"Dimension({self.vector.tolist()})"
        
    def to_string(self) -> str:
        """Provides a human-readable string representation."""
        if self.is_dimensionless():
            return "Dimensionless"
        
        parts = []
        for i, exp in enumerate(self.vector):
            if exp != 0:
                base_unit = BASE_DIMENSIONS[i]
                if exp == 1:
                    parts.append(base_unit)
                else:
                    # Format float exponents nicely
                    exp_str = f"{exp:.2f}".rstrip('0').rstrip('.')
                    parts.append(f"{base_unit}^{{{exp_str}}}")
        return " * ".join(parts)

# Predefined common dimensions
DIMENSIONLESS = Dimension([0] * len(BASE_DIMENSIONS))
