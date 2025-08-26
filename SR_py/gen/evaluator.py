# -*- coding: utf-8 -*-
"""
Evaluates symbolic expressions to numerical vectors.
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from ..dsl.expr import Expr
from ..config import CLIP_MIN, CLIP_MAX

logger = logging.getLogger(__name__)

class FeatureEvaluator:
    """
    Evaluates a list of features (expressions) on a given dataset.
    """
    def __init__(self,
                 clip_min: float = CLIP_MIN,
                 clip_max: float = CLIP_MAX,
                 nan_policy: str = 'omit'):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.nan_policy = nan_policy # 'omit' or 'raise'

    def evaluate(self,
                 features: List[Expr],
                 data: pd.DataFrame) -> (pd.DataFrame, List[Expr]):
        """
        Evaluate all features and return a DataFrame of numerical values.

        Args:
            features (List[Expr]): A list of symbolic features to evaluate.
            data (pd.DataFrame): The input data with columns matching variable names.

        Returns:
            pd.DataFrame: A dataframe where columns are the evaluated features.
            List[Expr]: The list of features that were successfully evaluated.
        """
        
        feature_vectors = {}
        valid_features = []
        data_dict = {col: data[col].values for col in data.columns}

        logger.info(f"Evaluating {len(features)} features...")
        
        for expr in tqdm(features, desc="Evaluating Features"):
            sig = expr.get_signature()
            try:
                with np.errstate(all='ignore'): # Suppress warnings for now
                    vec = expr.evaluate(data_dict)

                # --- Stability and Validity Checks ---
                # 1. Check for non-finite values
                if not np.all(np.isfinite(vec)):
                    if self.nan_policy == 'raise':
                        raise ValueError(f"Feature '{sig}' resulted in non-finite values (NaN/Inf).")
                    logger.debug(f"Skipping feature '{sig}' due to non-finite values.")
                    continue

                # 2. Clip values to a reasonable range
                vec = np.clip(vec, self.clip_min, self.clip_max)
                
                # 3. Check for constant features (zero variance)
                if np.std(vec) < 1e-8:
                    logger.debug(f"Skipping feature '{sig}' because it is constant.")
                    continue
                
                feature_vectors[sig] = vec
                valid_features.append(expr)

            except Exception as e:
                logger.error(f"Could not evaluate feature '{sig}'. Error: {e}", exc_info=False)

        logger.info(f"Successfully evaluated {len(valid_features)} out of {len(features)} features.")
        
        if not feature_vectors:
            return pd.DataFrame(), []
            
        return pd.DataFrame(feature_vectors), valid_features
