# -*- coding: utf-8 -*-
"""
The main SISSO Regressor pipeline.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Tuple, Callable, Dict, Any

from ..dsl.expr import Var, Expr
from ..dsl.dimension import Dimension
from ..gen.generator import FeatureGenerator
from ..gen.evaluator import FeatureEvaluator
from ..sis.screening import SIS
from ..sis.so import SparsifyingOperator
from ..config import DEFAULT_OPERATORS, RANDOM_STATE
from .report import build_report

logger = logging.getLogger(__name__)

class SissoRegressor:
    """
    Main class for the Sure Independence Screening and Sparsifying Operator (SISSO) regressor.
    """
    def __init__(self,
                 K: int = 2,
                 operators: List[Union[str, Callable, Tuple]] = DEFAULT_OPERATORS,
                 sis_screener: str = 'pearson',
                 sis_topk: int = 2000,
                 so_solver: str = 'omp',
                 so_max_terms: int = 3,
                 cv: int = 5,
                 dimensional_check: bool = False,
                 random_state: int = RANDOM_STATE,
                 n_jobs: int = -1):
        
        self.K = K
        self.operators = operators # This will need parsing for custom functions
        self.sis_screener = sis_screener
        self.sis_topk = sis_topk
        self.so_solver = so_solver
        self.so_max_terms = so_max_terms
        self.cv = cv
        self.dimensional_check = dimensional_check
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Internal components
        self.generator: Optional[FeatureGenerator] = None
        self.evaluator: Optional[FeatureEvaluator] = None
        self.screener: Optional[SIS] = None
        self.solver: Optional[SparsifyingOperator] = None
        
        # Results
        self.final_model_info: Dict = {}
        self.feature_space_: List[Expr] = []
        self.screened_features_: pd.DataFrame = pd.DataFrame()

    def _initialize_components(self):
        """Initialize all the internal components based on the parameters."""
        # TODO: Add logic to parse self.operators and register custom functions
        
        self.generator = FeatureGenerator(operators=[op for op in self.operators if isinstance(op, str)], 
                                          max_complexity=self.K * 3,
                                          dimensional_check=self.dimensional_check) # Pass the flag
        self.evaluator = FeatureEvaluator()
        self.screener = SIS(screener=self.sis_screener, top_k=self.sis_topk)
        self.solver = SparsifyingOperator(solver=self.so_solver, max_terms=self.so_max_terms, cv=self.cv)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            feature_dimensions: Optional[Dict[str, Dimension]] = None,
            target_dimension: Optional[Dimension] = None):
        """
        Fit the SISSO model.
        """
        logger.info("Starting SISSO regressor fit process...")
        self._initialize_components()
        
        # 1. Generate feature space
        if self.dimensional_check and feature_dimensions is None:
            raise ValueError("`feature_dimensions` must be provided when dimensional_check is True.")
            
        initial_features = [
            Var(name, dimension=feature_dimensions.get(name))
            if feature_dimensions else Var(name)
            for name in X.columns
        ]

        feature_layers = self.generator.generate(initial_features, self.K)
        self.feature_space_ = [expr for layer in feature_layers for expr in layer]
        
        # Optional: Filter features by target dimension
        if self.dimensional_check and target_dimension:
            logger.info(f"Filtering features to match target dimension: {target_dimension.to_string()}")
            original_count = len(self.feature_space_)
            self.feature_space_ = [
                f for f in self.feature_space_ if f.dimension == target_dimension
            ]
            logger.info(f"Retained {len(self.feature_space_)}/{original_count} features with correct dimension.")
            
        # 2. Evaluate features
        evaluated_features_df, valid_features = self.evaluator.evaluate(self.feature_space_, X)
        
        # 3. Screen features (SIS)
        self.screened_features_ = self.screener.screen(evaluated_features_df, y)
        
        # 4. Build sparse model (SO)
        self.solver.fit(self.screened_features_, y)
        self.final_model_info = self.solver.get_model_info()
        
        # 保存训练数据用于后续指标计算
        self._train_X = X.copy()
        self._train_y = y.copy()
        
        logger.info("SISSO regressor fit process completed.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the fitted SISSO model.
        """
        if not self.solver or not self.final_model_info:
            raise RuntimeError("Model has not been fitted yet.")
            
        # We need to evaluate the selected features on the new data X
        selected_feature_names = self.final_model_info.get('selected_features', [])
        
        # This is inefficient, should map names back to expr objects
        selected_exprs = [expr for expr in self.feature_space_ if expr.get_signature() in selected_feature_names]
        
        X_eval, _ = self.evaluator.evaluate(selected_exprs, X)
        
        return self.solver.predict(X_eval)

    def explain(self) -> Dict:
        """
        Return a dictionary with the final model, formula, and metrics.
        """
        # This is a placeholder for the final reporting logic
        return build_report(self)
