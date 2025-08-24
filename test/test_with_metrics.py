#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示SISSO-Py的自动性能指标计算功能
"""

import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

def main():
    print("🎯 SISSO-Py 自动性能指标演示")
    print("=" * 60)
    
    # 创建简单的测试数据
    np.random.seed(42)
    n = 200
    
    X = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n)
    })
    
    # 简单线性关系：y = 2*x1 + 3*x2 - x3
    y = 2*X['x1'] + 3*X['x2'] - X['x3'] + np.random.randn(n) * 0.1
    
    print(f"📊 数据: {n}个样本")
    print(f"📈 真实关系: y = 2×x1 + 3×x2 - x3 + noise")
    print()
    
    # 创建SISSO模型
    model = SissoRegressor(
        K=1,  # 只用一层，找线性关系
        operators=['+', '-', '*'],  # 基本操作符
        sis_screener='pearson',
        sis_topk=50,
        so_solver='omp',
        so_max_terms=3,
        cv=3,
        random_state=42
    )
    
    print("⚙️ 训练SISSO模型...")
    model.fit(X, y)
    
    # 生成包含性能指标的报告
    report = model.explain()
    
    print("\n" + "="*60)
    print("📊 完整报告 (包含自动计算的性能指标)")
    print("="*60)
    print(report)
    
    print("\n" + "="*60)
    print("📈 单独获取性能指标")
    print("="*60)
    
    # 也可以直接访问指标数据
    metrics = report['results']['metrics']
    print("性能指标详情:")
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n✨ 现在report中自动包含了完整的性能评估！")

if __name__ == "__main__":
    main()
