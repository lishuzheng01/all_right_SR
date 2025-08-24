# -*- coding: utf-8 -*-
"""
Provides formatted and human-readable report output for SISSO results.
"""
import json
from typing import Dict, Any, List
from textwrap import dedent
import re


class SissoReport:
    """
    A formatted report class that provides beautiful, readable output.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def __getitem__(self, key):
        """支持字典式访问，保持向后兼容"""
        return self.data[key]
    
    def __contains__(self, key):
        """支持 'in' 操作符"""
        return key in self.data
    
    def get(self, key, default=None):
        """支持 .get() 方法"""
        return self.data.get(key, default)
        
    def __str__(self) -> str:
        """返回格式化的完整报告"""
        if "status" in self.data and "not fitted" in self.data["status"]:
            return "❌ SISSO模型尚未训练"
            
        return self._format_full_report()
    
    def __repr__(self) -> str:
        """返回格式化的完整报告"""
        return self.__str__()
    
    def _format_full_report(self) -> str:
        """格式化完整报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("📊 SISSO 符号回归分析报告")
        lines.append("=" * 80)
        
        # 配置信息
        lines.append(self._format_configuration())
        
        # 训练统计
        lines.append(self._format_run_statistics())
        
        # 发现的公式
        lines.append(self._format_formula())
        
        # 特征详情
        lines.append(self._format_features())
        
        # 性能指标
        lines.append(self._format_metrics())
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_configuration(self) -> str:
        """格式化配置信息"""
        config = self.data.get("configuration", {})
        
        lines = []
        lines.append("\n🔧 模型配置")
        lines.append("-" * 50)
        lines.append(f"  复杂度层数 (K): {config.get('K', 'N/A')}")
        lines.append(f"  SIS筛选方法: {config.get('sis_screener', 'N/A')}")
        lines.append(f"  SIS保留特征: {config.get('sis_topk', 'N/A')}")
        lines.append(f"  稀疏求解器: {config.get('so_solver', 'N/A')}")
        lines.append(f"  最大模型项数: {config.get('so_max_terms', 'N/A')}")
        
        operators = config.get('operators', [])
        if operators:
            lines.append(f"  操作符集合: {', '.join(operators)}")
        
        return "\n".join(lines)
    
    def _format_run_statistics(self) -> str:
        """格式化运行统计"""
        run_info = self.data.get("run_info", {})
        
        lines = []
        lines.append("\n📈 训练统计")
        lines.append("-" * 50)
        lines.append(f"  生成特征总数: {run_info.get('total_features_generated', 'N/A'):,}")
        lines.append(f"  SIS筛选后特征: {run_info.get('features_after_sis', 'N/A'):,}")
        lines.append(f"  最终模型特征: {run_info.get('features_in_final_model', 'N/A')}")
        
        return "\n".join(lines)
    
    def _format_formula(self) -> str:
        """格式化发现的公式"""
        final_model = self.data.get("results", {}).get("final_model", {})
        
        lines = []
        lines.append("\n🎯 发现的符号公式")
        lines.append("-" * 50)
        
        # 显示美观的数学公式
        formula_latex = final_model.get("formula_latex", "")
        if formula_latex:
            readable_formula = self._make_formula_readable(formula_latex)
            lines.append(f"  数学表达式:")
            lines.append(f"    {readable_formula}")
            
        # 显示LaTeX代码（可选）
        lines.append(f"\n  LaTeX代码:")
        lines.append(f"    {formula_latex}")
        
        return "\n".join(lines)
    
    def _format_features(self) -> str:
        """格式化特征详情"""
        features = self.data.get("results", {}).get("final_model", {}).get("features", [])
        
        lines = []
        lines.append("\n🧮 模型特征详情")
        lines.append("-" * 50)
        
        if not features:
            lines.append("  暂无特征信息")
            return "\n".join(lines)
        
        # 表头
        lines.append(f"  {'序号':<4} {'系数':<12} {'复杂度':<6} {'特征表达式'}")
        lines.append(f"  {'-'*4} {'-'*12} {'-'*6} {'-'*30}")
        
        # 特征列表
        for i, feature in enumerate(features, 1):
            coeff = feature.get('coefficient', 0)
            complexity = feature.get('complexity', 0)
            signature = feature.get('signature', '')
            
            # 美化特征表达式
            readable_expr = self._make_expression_readable(signature)
            
            lines.append(f"  {i:<4} {coeff:<+12.4f} {complexity:<6} {readable_expr}")
        
        return "\n".join(lines)
    
    def _format_metrics(self) -> str:
        """格式化性能指标"""
        metrics = self.data.get("results", {}).get("metrics", {})
        
        lines = []
        lines.append("\n📊 模型性能指标")
        lines.append("-" * 50)
        
        # 检查是否有训练集指标
        train_rmse = metrics.get("train_rmse")
        train_mae = metrics.get("train_mae")
        train_r2 = metrics.get("train_r2")
        train_samples = metrics.get("train_samples")
        error_msg = metrics.get("error")
        note_msg = metrics.get("note")
        
        if error_msg:
            lines.append(f"  ❌ 指标计算错误: {error_msg}")
        elif note_msg:
            lines.append(f"  ℹ️  {note_msg}")
        elif train_rmse is not None:
            lines.append(f"  📈 训练集性能 (基于 {train_samples} 个样本):")
            lines.append(f"    RMSE (均方根误差): {train_rmse:.6f}")
            if train_mae is not None:
                lines.append(f"    MAE  (平均绝对误差): {train_mae:.6f}")
            if train_r2 is not None:
                lines.append(f"    R²   (决定系数):    {train_r2:.6f}")
                
            # 添加性能解释
            if train_r2 is not None:
                if train_r2 >= 0.95:
                    performance = "优秀"
                elif train_r2 >= 0.90:
                    performance = "良好"
                elif train_r2 >= 0.80:
                    performance = "中等"
                elif train_r2 >= 0.60:
                    performance = "一般"
                else:
                    performance = "需要改进"
                lines.append(f"    模型拟合质量: {performance}")
        else:
            lines.append("  性能指标计算中...")
        
        # 检查是否有测试集指标
        test_rmse = metrics.get("test_rmse")
        test_mae = metrics.get("test_mae")
        test_r2 = metrics.get("test_r2")
        
        if test_rmse is not None:
            lines.append(f"  📊 测试集性能:")
            lines.append(f"    RMSE: {test_rmse:.6f}")
            if test_mae is not None:
                lines.append(f"    MAE:  {test_mae:.6f}")
            if test_r2 is not None:
                lines.append(f"    R²:   {test_r2:.6f}")
        
        return "\n".join(lines)
    
    def _make_formula_readable(self, latex_formula: str) -> str:
        """将LaTeX公式转换为更易读的数学表达式"""
        if not latex_formula:
            return ""
            
        # 移除LaTeX命令并转换为更易读的格式
        formula = latex_formula
        
        # 替换LaTeX数学符号
        formula = formula.replace("\\times", " × ")
        formula = formula.replace("\\cdot", " × ")
        formula = formula.replace("\\frac{", "(")
        formula = formula.replace("}{", ")/(")
        
        # 处理上标 - 保持花括号内的内容
        formula = re.sub(r'\^\{(\d+)\}', r'^\1', formula)
        formula = re.sub(r'\^(\d)', r'^\1', formula)
        
        # 简化多层括号
        # 重复应用简化，直到没有变化
        prev_formula = ""
        while prev_formula != formula:
            prev_formula = formula
            formula = re.sub(r'\(\(([^()]+)\)\)', r'(\1)', formula)
        
        # 美化空格
        formula = re.sub(r'\s+', ' ', formula)  # 多个空格合并为一个
        formula = formula.strip()
        
        return formula
    
    def _make_expression_readable(self, signature: str) -> str:
        """将特征签名转换为更易读的表达式"""
        expr = signature
        
        # 替换函数名为更直观的符号
        expr = expr.replace("square(", "square(")  # 保持原样，稍后处理
        expr = expr.replace("sqrt(", "√(")
        expr = expr.replace("log(", "ln(")
        
        # 处理二元运算符
        expr = expr.replace("+(", "(")
        expr = expr.replace("-(", "(") 
        expr = expr.replace("*(", "(")
        expr = expr.replace("safe_div(", "(")
        
        # 替换逗号为相应运算符
        # 需要根据前面的操作符类型来决定
        parts = signature.split('(')
        if len(parts) > 1:
            # 简化显示
            if signature.startswith('square('):
                inner = signature[7:-1]  # 去掉 'square(' 和 ')'
                return f"({self._make_expression_readable(inner)})²"
            elif '+(' in signature:
                # 处理加法
                content = signature[signature.find('(')+1:signature.rfind(')')]
                if ',' in content:
                    left, right = content.split(',', 1)
                    return f"{self._make_expression_readable(left)} + {self._make_expression_readable(right)}"
            elif '-(' in signature:
                # 处理减法
                content = signature[signature.find('(')+1:signature.rfind(')')]
                if ',' in content:
                    left, right = content.split(',', 1)
                    return f"{self._make_expression_readable(left)} - {self._make_expression_readable(right)}"
            elif '*(' in signature:
                # 处理乘法
                content = signature[signature.find('(')+1:signature.rfind(')')]
                if ',' in content:
                    left, right = content.split(',', 1)
                    return f"{self._make_expression_readable(left)} × {self._make_expression_readable(right)}"
        
        # 如果是变量，直接返回
        if signature.startswith('x'):
            return signature
            
        return signature
    
    def get_formula(self, format='readable') -> str:
        """
        获取公式的指定格式
        
        Args:
            format: 'readable' (易读格式), 'latex' (LaTeX格式), 'sympy' (SymPy格式)
        """
        final_model = self.data.get("results", {}).get("final_model", {})
        
        if format == 'readable':
            latex = final_model.get("formula_latex", "")
            return self._make_formula_readable(latex)
        elif format == 'latex':
            return final_model.get("formula_latex", "")
        elif format == 'sympy':
            return final_model.get("formula_sympy", "")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_features_table(self) -> str:
        """获取格式化的特征表格"""
        return self._format_features()
    
    def to_json(self, indent=2) -> str:
        """返回JSON格式的报告"""
        return json.dumps(self.data, indent=indent, ensure_ascii=False)
