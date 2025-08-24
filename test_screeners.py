#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示SISSO-Py的多种筛选方法
"""

import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

def main():
    print("🔍 SISSO-Py 多种筛选方法演示")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n = 300
    
    X = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'x3': np.random.randn(n),
        'x4': np.random.randn(n)
    })
    
    # 目标函数：y = 2*x1 + x2^2 - x3 + noise
    # x4 是噪声变量，不应该被选中
    y = 2*X['x1'] + X['x2']**2 - X['x3'] + np.random.randn(n) * 0.1
    
    print(f"📊 数据: {n}个样本, 4个特征")
    print(f"📈 真实关系: y = 2×x1 + x2² - x3 + noise")
    print(f"📌 x4 是无关噪声变量")
    print()
    
    # 测试所有可用的筛选方法
    screener_methods = [
        'pearson',      # Pearson相关系数
        'mutual_info',  # 互信息
        'random',       # 随机筛选
        'variance',     # 方差筛选
        'f_regression', # F统计量
        'rfe',          # 递归特征消除
        'lasso_path',   # LASSO路径
        'combined'      # 组合方法
    ]
    
    results = {}
    
    for screener in screener_methods:
        print(f"🔍 测试筛选方法: {screener}")
        print("-" * 30)
        
        try:
            model = SissoRegressor(
                K=1,                    # 简单模型
                operators=['+', '-', '*', 'square'],
                sis_screener=screener,  # 使用不同的筛选方法
                sis_topk=3,            # 只选择前3个特征
                so_solver='omp',
                so_max_terms=3,
                cv=3,
                random_state=42
            )
            
            model.fit(X, y)
            report = model.explain()
            
            # 提取性能指标
            metrics = report['results']['metrics']
            r2 = metrics.get('train_r2', 0)
            rmse = metrics.get('train_rmse', float('inf'))
            
            # 提取选中的特征
            features = report['results']['final_model']['features']
            selected_vars = [f['signature'] for f in features]
            
            results[screener] = {
                'r2': r2,
                'rmse': rmse,
                'features': selected_vars[:3]  # 只看前3个
            }
            
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  选中特征: {', '.join(selected_vars)}")
            print()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results[screener] = {'error': str(e)}
            print()
    
    # 总结结果
    print("=" * 60)
    print("📋 筛选方法性能对比")
    print("=" * 60)
    
    print(f"{'方法':<12} {'R²':<8} {'RMSE':<8} 主要选中特征")
    print("-" * 60)
    
    for method, result in results.items():
        if 'error' in result:
            print(f"{method:<12} {'ERROR':<8} {'ERROR':<8} {result['error'][:20]}...")
        else:
            r2 = result['r2']
            rmse = result['rmse']
            features = ', '.join(result['features'][:2]) + '...' if len(result['features']) > 2 else ', '.join(result['features'])
            print(f"{method:<12} {r2:<8.4f} {rmse:<8.4f} {features}")
    
    print("\n🎯 筛选方法说明:")
    print("  • pearson    - 基于Pearson相关系数")
    print("  • mutual_info - 基于互信息")
    print("  • random     - 随机选择特征")
    print("  • variance   - 基于特征方差")
    print("  • f_regression - 基于F统计量")
    print("  • rfe        - 递归特征消除")
    print("  • lasso_path - 基于LASSO正则化路径")
    print("  • combined   - 多方法投票组合")
    
    print(f"\n✨ 现在SISSO-Py支持{len(screener_methods)}种不同的特征筛选方法！")

if __name__ == "__main__":
    main()
