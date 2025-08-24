#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SISSO-Py 筛选方法完整演示
展示所有8种特征筛选方法的使用和特点
"""

import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

def demonstrate_screener_methods():
    """演示所有可用的筛选方法"""
    
    print("🔍 SISSO-Py 特征筛选方法完整指南")
    print("=" * 80)
    
    # 创建有明确模式的测试数据
    np.random.seed(123)
    n = 500
    
    X = pd.DataFrame({
        'signal1': np.random.randn(n),      # 重要信号
        'signal2': np.random.randn(n),      # 重要信号  
        'noise1': np.random.randn(n),       # 噪声
        'noise2': np.random.randn(n),       # 噪声
        'correlated': None                  # 与目标相关但非因果
    })
    
    # 目标：y = 3*signal1 + 2*signal2^2 + noise
    y = 3*X['signal1'] + 2*X['signal2']**2 + np.random.randn(n) * 0.2
    X['correlated'] = y + np.random.randn(n) * 0.5  # 添加与目标相关的变量
    
    print(f"📊 测试数据设置:")
    print(f"  样本数: {n}")
    print(f"  真实模型: y = 3×signal1 + 2×signal2² + noise")
    print(f"  特征说明:")
    print(f"    • signal1, signal2: 真实的因果特征")
    print(f"    • noise1, noise2: 随机噪声特征")
    print(f"    • correlated: 与目标相关但非因果的特征")
    print()
    
    # 所有筛选方法详细说明
    screener_info = {
        'pearson': {
            'name': 'Pearson相关系数',
            'description': '基于线性相关性，适合线性关系',
            'pros': '简单快速，解释性强',
            'cons': '只能捕获线性关系',
            'best_for': '线性模型、预探索'
        },
        'mutual_info': {
            'name': '互信息',
            'description': '基于信息理论，能捕获非线性关系',
            'pros': '能发现复杂非线性关系',
            'cons': '计算较慢，可能过拟合',
            'best_for': '非线性关系、复杂模式'
        },
        'random': {
            'name': '随机筛选',
            'description': '随机选择特征，作为基线对比',
            'pros': '无偏差，适合基线测试',
            'cons': '可能选到无关特征',
            'best_for': '基线对比、随机搜索'
        },
        'variance': {
            'name': '方差筛选', 
            'description': '选择方差大的特征，去除常数特征',
            'pros': '快速去除无变化特征',
            'cons': '忽略与目标的关系',
            'best_for': '预处理、去除常数特征'
        },
        'f_regression': {
            'name': 'F统计量',
            'description': '基于单变量线性回归的F统计量',
            'pros': '统计学基础，标准方法',
            'cons': '假设线性关系',
            'best_for': '统计建模、线性关系'
        },
        'rfe': {
            'name': '递归特征消除',
            'description': '递归训练模型并消除最不重要特征',
            'pros': '考虑特征间交互，精确',
            'cons': '计算成本高',
            'best_for': '精确建模、小特征集'
        },
        'lasso_path': {
            'name': 'LASSO路径',
            'description': '基于LASSO正则化路径的特征选择',
            'pros': '自动特征选择，处理共线性',
            'cons': '可能选择共线特征中的任意一个',
            'best_for': '高维数据、稀疏模型'
        },
        'combined': {
            'name': '组合投票',
            'description': '多种方法投票决定，综合各方法优势',
            'pros': '鲁棒性强，综合多种视角',
            'cons': '计算成本高',
            'best_for': '重要项目、追求稳定性'
        }
    }
    
    print("📋 筛选方法详细说明:")
    print("=" * 80)
    for method, info in screener_info.items():
        print(f"\n🔸 {method.upper()}: {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   优点: {info['pros']}")
        print(f"   缺点: {info['cons']}")
        print(f"   适用: {info['best_for']}")
    
    print("\n" + "=" * 80)
    print("🧪 实际性能测试 (选择前3个特征)")
    print("=" * 80)
    
    results = {}
    
    for method in screener_info.keys():
        print(f"\n🔍 测试 {method.upper()}...")
        
        try:
            model = SissoRegressor(
                K=1,  # 简单模型
                operators=['+', '-', '*', 'square'],
                sis_screener=method,
                sis_topk=3,
                so_solver='omp', 
                so_max_terms=3,
                cv=3,
                random_state=123
            )
            
            model.fit(X, y)
            report = model.explain()
            
            # 分析结果
            r2 = report['results']['metrics']['train_r2']
            rmse = report['results']['metrics']['train_rmse']
            features = [f['signature'] for f in report['results']['final_model']['features']]
            
            # 分析特征质量
            feature_quality = analyze_feature_selection(features, ['signal1', 'signal2'])
            
            results[method] = {
                'r2': r2,
                'rmse': rmse,
                'features': features,
                'quality': feature_quality
            }
            
            print(f"   R² = {r2:.4f}, RMSE = {rmse:.4f}")
            print(f"   选中特征: {', '.join(features[:3])}")
            print(f"   特征质量: {feature_quality}")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            results[method] = {'error': str(e)}
    
    # 总结和推荐
    print("\n" + "=" * 80)
    print("📊 性能总结和使用建议")
    print("=" * 80)
    
    # 按R²排序
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print(f"\n🏆 性能排名 (按R²):")
    for i, (method, result) in enumerate(sorted_results, 1):
        quality_icon = "🎯" if "good" in result['quality'] else "⚠️" if "mixed" in result['quality'] else "❌"
        print(f"   {i}. {method:<12} R²={result['r2']:.4f} {quality_icon} {result['quality']}")
    
    print(f"\n💡 使用建议:")
    print(f"   • 探索阶段: 使用 'pearson' 或 'mutual_info' 快速了解数据")
    print(f"   • 线性关系: 使用 'pearson' 或 'f_regression'")
    print(f"   • 非线性关系: 使用 'mutual_info' 或 'lasso_path'")
    print(f"   • 高维数据: 使用 'lasso_path' 或 'variance'")
    print(f"   • 稳健建模: 使用 'combined' 或 'rfe'")
    print(f"   • 基线对比: 使用 'random'")
    
    return results

def analyze_feature_selection(selected_features, true_features):
    """分析特征选择质量"""
    # 简单的启发式分析
    signal_count = sum(1 for f in selected_features if any(tf in f for tf in true_features))
    noise_count = len(selected_features) - signal_count
    
    if signal_count >= 2:
        return "good - 找到主要信号特征"
    elif signal_count == 1:
        return "mixed - 找到部分信号特征"
    else:
        return "poor - 未找到信号特征"

if __name__ == "__main__":
    demonstrate_screener_methods()
