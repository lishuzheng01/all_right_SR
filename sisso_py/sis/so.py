# -*- coding: utf-8 -*-
"""
Sparsifying Operator (SO) methods for model building.
"""
from typing import Dict, Type
import pandas as pd
import logging
from sklearn.linear_model import (
    OrthogonalMatchingPursuit,
    Lasso,
    ElasticNet,
    LinearRegression
)
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score
import numpy as np

logger = logging.getLogger(__name__)

# Registry for different sparse solvers
SOLVERS: Dict[str, Type[RegressorMixin]] = {
    'omp': OrthogonalMatchingPursuit,
    'lasso': Lasso,
    'elasticnet': ElasticNet,
}

class SparsifyingOperator:
    """
    Builds a sparse linear model from the screened feature set.
    """
    def __init__(self,
                 solver: str = 'omp',
                 max_terms: int = 3,
                 cv: int = 5,
                 **solver_kwargs):
        
        if solver not in SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'. "
                             f"Available options: {list(SOLVERS.keys())}")
        
        self.solver_name = solver
        # OMP uses n_nonzero_coefs, others might use alpha.
        # This is a simplified way to handle it.
        if solver == 'omp':
            solver_kwargs.setdefault('n_nonzero_coefs', max_terms)
        else:
            solver_kwargs.setdefault('alpha', 0.1)

        self.solver = SOLVERS[solver](**solver_kwargs)
        self.cv = cv
        self.final_model = None # Will be a simple LinearRegression on selected features
        self.selected_features_ = []

        logger.info(f"SparsifyingOperator initialized with solver='{solver}' and max_terms={max_terms}.")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the sparse model to find the most important features.

        Args:
            X (pd.DataFrame): The feature matrix from SIS.
            y (pd.Series): The target vector.
        """
        logger.info(f"Fitting sparse model on {X.shape[1]} features...")
        
        if X.empty:
            logger.warning("Input feature matrix is empty. Cannot fit SparsifyingOperator.")
            return

        self.solver.fit(X, y)
        
        # Identify selected features
        if hasattr(self.solver, 'coef_'):
            coefs = pd.Series(self.solver.coef_, index=X.columns)
            self.selected_features_ = list(coefs[coefs.abs() > 1e-8].index)
        else:
             # For models like OMP after fitting
            logger.warning("Solver does not have 'coef_'. This path is not fully implemented.")
            self.selected_features_ = []

        if not self.selected_features_:
            logger.warning("The sparse solver did not select any features.")
            return

        logger.info(f"Sparse solver selected {len(self.selected_features_)} features: {self.selected_features_}")
        
        # Refit a simple linear model on the selected features
        self.final_model = LinearRegression()
        self.final_model.fit(X[self.selected_features_], y)
        
        # Evaluate model using cross-validation
        if self.cv > 1:
            scores = cross_val_score(self.final_model, X[self.selected_features_], y, cv=self.cv, scoring='neg_root_mean_squared_error')
            logger.info(f"Cross-validation RMSE: {-np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the final fitted model.
        """
        if self.final_model is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        return self.final_model.predict(X[self.selected_features_])

    def get_model_info(self) -> Dict:
        """
        Returns information about the final model.
        """
        if self.final_model is None:
            return {}
        
        return {
            'selected_features': self.selected_features_,
            'coefficients': dict(zip(self.selected_features_, self.final_model.coef_)),
            'intercept': self.final_model.intercept_,
        }
