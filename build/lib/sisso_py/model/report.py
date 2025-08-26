# -*- coding: utf-8 -*-
"""
Generates reports for the SISSO model results.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .formatted_report import SissoReport

def build_report(model) -> SissoReport:
    """
    Builds a comprehensive report from a fitted SissoRegressor instance.
    """
    if not model.solver or not model.final_model_info:
        return SissoReport({"status": "Model not fitted."})

    # Find the expression objects corresponding to the selected features
    sig_to_expr = {expr.get_signature(): expr for expr in model.feature_space_}
    
    selected_features_info = []
    final_formula_latex = f"{model.final_model_info.get('intercept', 0):.4g}"
    final_formula_sympy = model.final_model_info.get('intercept', 0)

    coefficients = model.final_model_info.get('coefficients', {})
    for name, coeff in coefficients.items():
        expr = sig_to_expr.get(name)
        if expr:
            selected_features_info.append({
                'signature': name,
                'latex': expr.to_latex(),
                'complexity': expr.get_complexity(),
                'coefficient': coeff,
            })
            
            # Build final formula string  
            if coeff >= 0:
                sign = "+" if final_formula_latex else ""
                term_latex = f"{sign}{coeff:.4g} \\times {expr.to_latex()}"
            else:
                term_latex = f"{coeff:.4g} \\times {expr.to_latex()}"
            final_formula_latex += f" {term_latex}"
            final_formula_sympy += coeff * expr.to_sympy()

    # 计算训练集性能指标
    train_metrics = {}
    if hasattr(model, '_train_X') and hasattr(model, '_train_y'):
        try:
            y_train_pred = model.predict(model._train_X)
            mse = mean_squared_error(model._train_y, y_train_pred)
            train_metrics = {
                "train_mse": mse,
                "train_rmse": np.sqrt(mse),
                "train_mae": mean_absolute_error(model._train_y, y_train_pred),
                "train_r2": r2_score(model._train_y, y_train_pred),
                "train_samples": len(model._train_y)
            }
        except Exception as e:
            train_metrics = {
                "train_mse": None,
                "train_rmse": None,
                "train_mae": None,
                "train_r2": None,
                "error": str(e)
            }
    else:
        train_metrics = {
            "train_mse": None,
            "train_rmse": None,
            "train_mae": None,
            "train_r2": None,
            "note": "Training data not available for metrics calculation"
        }

    report = {
        "configuration": {
            "K": model.K,
            "operators": [op for op in model.operators if isinstance(op, str)], # Simplified
            "sis_screener": model.sis_screener,
            "sis_topk": model.sis_topk,
            "so_solver": model.so_solver,
            "so_max_terms": model.so_max_terms,
        },
        "results": {
            "final_model": {
                "formula_latex": final_formula_latex,
                "formula_sympy": str(final_formula_sympy),
                "intercept": model.final_model_info.get('intercept'),
                "features": selected_features_info
            },
            # 自动计算的性能指标
            "metrics": train_metrics
        },
        "run_info": {
            "total_features_generated": len(model.feature_space_),
            "features_after_sis": model.screened_features_.shape[1],
            "features_in_final_model": len(coefficients),
        }
    }
    
    return SissoReport(report)
