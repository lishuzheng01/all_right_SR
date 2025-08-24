#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SISSO-Py 美观报告演示
展示新的格式化输出功能
"""

import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

def main():
    print("🎨 SISSO-Py 美观报告演示")
    print("=" * 60)
    
    # 生成简单的测试数据
    np.random.seed(123)
    n = 200
    
    X = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n)
    })
    
    # 简单的二次关系：y = x1² + 2*x2 - x3
    y = X['x1']**2 + 2*X['x2'] - X['x3'] + np.random.randn(n) * 0.1
    
    print(f"📊 数据: {n}个样本, 真实关系: y = x1² + 2×x2 - x3 + noise")
    print()
    
    # 创建SISSO模型
    model = SissoRegressor(
        K=2,
        operators=['+', '-', '*', 'square'],
        sis_screener='pearson',
        sis_topk=100,
        so_solver='omp',
        so_max_terms=3,
        cv=3,
        random_state=123
    )
    
    print("⚙️ 训练SISSO模型...")
    model.fit(X, y)
    
    # 生成美观的报告
    report = model.explain()
    
    print("\n" + "="*60)
    print("📋 方法1: 直接打印报告 (推荐)")
    print("="*60)
    print("使用: print(report)")
    print("-" * 60)
    print(report)
    
    print("\n" + "="*60)
    print("📋 方法2: 获取特定格式的公式")
    print("="*60)
    
    print("\n🔤 易读格式:")
    print(f"  {report.get_formula('readable')}")
    
    print("\n📐 LaTeX格式 (用于论文):")
    print(f"  {report.get_formula('latex')}")
    
    print("\n🐍 SymPy格式 (用于计算):")
    print(f"  {report.get_formula('sympy')}")
    
    print("\n" + "="*60)
    print("📋 方法3: 获取JSON格式 (用于数据交换)")
    print("="*60)
    json_report = report.to_json(indent=2)
    print("前100个字符:", json_report[:100] + "...")
    
    print(f"\n✨ 完成！SISSO-Py现在支持美观、清晰的报告输出")

if __name__ == "__main__":
    main()
