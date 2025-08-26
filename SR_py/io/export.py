# -*- coding: utf-8 -*-
"""
Utilities for exporting formulas to different formats.
"""
import sympy
import json

from ..model.report import build_report

def export_to_latex(model) -> str:
    """Exports the final model formula to a LaTeX string."""
    report = build_report(model)
    return report.get('results', {}).get('final_model', {}).get('formula_latex', 'Model not available.')

def export_to_sympy(model):
    """Exports the final model formula to a SymPy object."""
    report = build_report(model)
    formula_str = report.get('results', {}).get('final_model', {}).get('formula_sympy', '0')
    return sympy.sympify(formula_str)

def export_to_json(model, filepath: str):
    """Exports the full model report to a JSON file."""
    report = build_report(model)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
