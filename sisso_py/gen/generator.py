# -*- coding: utf-8 -*-
"""
Generates the feature space by applying operators to existing features.
"""
from typing import List, Set, Dict
from itertools import product, combinations_with_replacement
import logging

from ..dsl.expr import Expr, Var, Unary, Binary
from ..dsl.complexity import ComplexityBudget
from ..ops.base import get_all_operators, Operator

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Generates new features layer by layer, up to a specified complexity.
    """
    def __init__(self,
                 operators: List[str],
                 max_complexity: int,
                 dimensional_check: bool = False):
        
        self.all_ops: Dict[str, Operator] = {k: v for k, v in get_all_operators().items() if k in operators}
        self.unary_ops: List[Operator] = [op for op in self.all_ops.values() if op.arity == 1]
        self.binary_ops: List[Operator] = [op for op in self.all_ops.values() if op.arity == 2]
        self.budget = ComplexityBudget(max_total_complexity=max_complexity)
        self.dimensional_check = dimensional_check
        
        logger.info(f"FeatureGenerator initialized with {len(self.unary_ops)} unary and {len(self.binary_ops)} binary operators.")
        logger.info(f"Unary operators: {[op.name for op in self.unary_ops]}")
        logger.info(f"Binary operators: {[op.name for op in self.binary_ops]}")
        logger.info(f"Max complexity set to: {max_complexity}")

    def generate(self, initial_features: List[Expr], n_layers: int) -> List[List[Expr]]:
        """
        Generate feature space for a given number of layers (K in the paper).

        Args:
            initial_features (List[Expr]): The base features (L0).
            n_layers (int): The number of layers to generate.

        Returns:
            List[List[Expr]]: A list where each element is a list of features 
                              for that layer.
        """
        
        feature_space: List[List[Expr]] = [initial_features]
        seen_signatures: Set[str] = {f.get_signature() for f in initial_features}

        for i in range(1, n_layers + 1):
            logger.info(f"--- Generating Layer {i} ---")
            
            new_features_layer: List[Expr] = []
            
            # Combine features from all previous layers
            candidates_from_previous_layers = [
                item for sublist in feature_space for item in sublist
            ]

            # 1. Apply unary operators
            for op in self.unary_ops:
                for f in candidates_from_previous_layers:
                    try:
                        new_expr = Unary(op.name, f)
                    except TypeError as e:
                        if self.dimensional_check:
                            logger.debug(f"Dimensional error creating Unary('{op.name}', '{f.get_signature()}'): {e}")
                            continue # Skip dimensionally invalid expressions
                        else:
                            raise e # Re-raise if not in dimensional check mode

                    if not self.budget.is_within_budget(new_expr.get_complexity()):
                        continue
                    
                    sig = new_expr.get_signature()
                    if sig not in seen_signatures:
                        new_features_layer.append(new_expr)
                        seen_signatures.add(sig)

            # 2. Apply binary operators
            # Combine features from all previous layers with each other
            feature_pairs = list(combinations_with_replacement(candidates_from_previous_layers, 2))
            
            for op in self.binary_ops:
                for f1, f2 in feature_pairs:
                    
                    # Avoid trivial combinations like f1+f1 unless it's a square or similar
                    if f1 == f2 and op.name in ['+', '*']: continue
                    
                    try:
                        new_expr = Binary(op.name, f1, f2)
                    except TypeError as e:
                        if self.dimensional_check:
                            logger.debug(f"Dimensional error creating Binary('{op.name}', '{f1.get_signature()}', '{f2.get_signature()}'): {e}")
                            continue
                        else:
                            raise e

                    if not self.budget.is_within_budget(new_expr.get_complexity()):
                        continue
                        
                    sig = new_expr.get_signature()
                    if sig not in seen_signatures:
                        new_features_layer.append(new_expr)
                        seen_signatures.add(sig)

            logger.info(f"Layer {i}: Generated {len(new_features_layer)} new features.")
            if not new_features_layer:
                logger.warning(f"No new features generated for layer {i}. Stopping generation.")
                break
                
            feature_space.append(new_features_layer)

        return feature_space
