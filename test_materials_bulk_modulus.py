#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
材料科学SISSO应用：二元化合物体积模量预测
场景：从元素属性（电负性、原子半径、价电子数等）推断化合物A_xB_y的体积模量K
目标：发现低维显式解析公式
"""

from turtle import clear
import numpy as np
import pandas as pd
from sisso_py import SissoRegressor
import matplotlib.pyplot as plt
import seaborn as sns



def load_element_properties():
    """加载元素属性数据"""
    # 常见元素的基本属性数据
    elements_data = {
        'Element': ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 
                   'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                   'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                   'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                   'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'Pb'],
        
        # 电负性 (Pauling scale)
        'Electronegativity': [0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.93, 1.31, 1.61,
                             1.90, 2.19, 2.58, 3.16, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66,
                             1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55,
                             2.96, 0.82, 0.95, 1.22, 1.33, 1.6, 2.16, 1.9, 2.2, 2.28,
                             2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.1, 2.66, 0.79, 0.89, 2.33],
        
        # 原子半径 (pm)
        'Atomic_Radius': [152, 112, 87, 67, 56, 48, 42, 186, 160, 118,
                         111, 98, 88, 79, 227, 197, 162, 147, 134, 128,
                         127, 126, 125, 124, 128, 134, 135, 122, 119, 120,
                         114, 248, 215, 180, 160, 146, 139, 136, 134, 134,
                         137, 144, 151, 167, 158, 133, 123, 115, 265, 222, 175],
        
        # 价电子数
        'Valence_Electrons': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
                             4, 5, 6, 7, 1, 2, 3, 4, 5, 6,
                             7, 8, 9, 10, 11, 12, 3, 4, 5, 6,
                             7, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 3, 4, 5, 6, 7, 1, 2, 4],
        
        # 原子质量 (u)
        'Atomic_Mass': [6.94, 9.01, 10.81, 12.01, 14.01, 16.00, 19.00, 22.99, 24.31, 26.98,
                       28.09, 30.97, 32.07, 35.45, 39.10, 40.08, 44.96, 47.87, 50.94, 52.00,
                       54.94, 55.85, 58.93, 58.69, 63.55, 65.38, 69.72, 72.64, 74.92, 78.96,
                       79.90, 85.47, 87.62, 88.91, 91.22, 92.91, 95.96, 98.91, 101.07, 102.91,
                       106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 132.91, 137.33, 207.2],
        
        # 第一电离能 (eV)
        'First_Ionization': [5.39, 9.32, 8.30, 11.26, 14.53, 13.62, 17.42, 5.14, 7.65, 5.99,
                            8.15, 10.49, 10.36, 12.97, 4.34, 6.11, 6.56, 6.83, 6.75, 6.77,
                            7.43, 7.90, 7.88, 7.64, 7.73, 9.39, 5.99, 7.90, 9.79, 9.75,
                            11.81, 4.18, 5.69, 6.22, 6.63, 6.76, 7.09, 7.28, 7.36, 7.46,
                            8.34, 7.58, 8.99, 5.79, 7.34, 8.61, 9.01, 10.45, 3.89, 5.21, 7.42]
    }
    
    return pd.DataFrame(elements_data)

def generate_binary_compounds_data():
    """生成二元化合物的体积模量数据"""
    # 基于文献数据的二元化合物体积模量，只包含元素表中存在的元素
    compounds_data = {
        'Compound': ['LiF', 'LiCl', 'NaCl', 'NaF', 'MgO', 'CaO', 
                    'Al2O3', 'SiO2', 'SiC', 'AlN', 
                    'ZnO', 'ZnS', 'CdS', 'CdTe', 'GaAs', 'InP', 'GaP', 'InAs',
                    'FeO', 'NiO', 'CoO', 'CuO', 'AgCl', 'CdCl2', 'PbS',
                    'SnTe', 'GeS', 'GeSe', 'SiS2', 'MoS2'],
        
        'Element_A': ['Li', 'Li', 'Na', 'Na', 'Mg', 'Ca',
                     'Al', 'Si', 'Si', 'Al',
                     'Zn', 'Zn', 'Cd', 'Cd', 'Ga', 'In', 'Ga', 'In',
                     'Fe', 'Ni', 'Co', 'Cu', 'Ag', 'Cd', 'Pb',
                     'Sn', 'Ge', 'Ge', 'Si', 'Mo'],
        
        'Element_B': ['F', 'Cl', 'Cl', 'F', 'O', 'O',
                     'O', 'O', 'C', 'N',
                     'O', 'S', 'S', 'Te', 'As', 'P', 'P', 'As',
                     'O', 'O', 'O', 'O', 'Cl', 'Cl', 'S',
                     'Te', 'S', 'Se', 'S', 'S'],
        
        # 体积模量 (GPa) - 来源于材料数据库和文献
        'Bulk_Modulus': [62, 24, 24, 47, 160, 117,
                        252, 37, 220, 208,
                        142, 77, 62, 43, 75, 71, 88, 58,
                        179, 180, 175, 140, 25, 18, 46,
                        43, 51, 49, 52, 198]
    }
    
    return pd.DataFrame(compounds_data)

def create_feature_matrix(compounds_df, elements_df):
    """创建特征矩阵"""
    features = []
    
    for _, compound in compounds_df.iterrows():
        elem_a = compound['Element_A']
        elem_b = compound['Element_B']
        
        # 获取元素A和B的属性
        props_a = elements_df[elements_df['Element'] == elem_a].iloc[0]
        props_b = elements_df[elements_df['Element'] == elem_b].iloc[0]
        
        # 创建特征向量
        feature_row = {
            # A元素属性
            'A_Electronegativity': props_a['Electronegativity'],
            'A_Atomic_Radius': props_a['Atomic_Radius'],
            'A_Valence_Electrons': props_a['Valence_Electrons'],
            'A_Atomic_Mass': props_a['Atomic_Mass'],
            'A_First_Ionization': props_a['First_Ionization'],
            
            # B元素属性
            'B_Electronegativity': props_b['Electronegativity'],
            'B_Atomic_Radius': props_b['Atomic_Radius'],
            'B_Valence_Electrons': props_b['Valence_Electrons'],
            'B_Atomic_Mass': props_b['Atomic_Mass'],
            'B_First_Ionization': props_b['First_Ionization'],
            
            # 目标变量
            'Bulk_Modulus': compound['Bulk_Modulus']
        }
        
        features.append(feature_row)
    
    return pd.DataFrame(features)

def analyze_correlations(df):
    """分析特征与体积模量的相关性"""
    print("🔍 特征与体积模量的相关性分析")
    print("=" * 60)
    
    # 计算相关系数
    target = df['Bulk_Modulus']
    feature_cols = [col for col in df.columns if col != 'Bulk_Modulus']
    
    correlations = []
    for col in feature_cols:
        corr = df[col].corr(target)
        correlations.append((col, corr))
    
    # 按相关性排序
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("特征相关性排名（按绝对值）:")
    for i, (feature, corr) in enumerate(correlations[:10], 1):
        print(f"{i:2d}. {feature:<20} {corr:+7.4f}")
    
    return correlations

def run_sisso_analysis(X, y):
    """运行SISSO分析"""
    print("\n🔬 SISSO符号回归分析")
    print("=" * 60)
    
    # 多种筛选方法对比
    screener_methods = ['mutual_info']
    results = {}
    
    for method in screener_methods:
        print(f"\n🔍 使用 {method} 筛选方法...")
        
        try:
            model = SissoRegressor(
                K=5,  # 复杂度层数
                operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log','exp','abs','reciprocal'],
                sis_screener=method,
                sis_topk=20,  # 筛选前20个特征
                so_solver='lasso',
                so_max_terms=2,  # 最大4项
                cv=5,
                random_state=42
            )
            
            model.fit(X, y)
            report = model.explain()
            
            # 提取关键信息
            metrics = report['results']['metrics']
            formula = report.get_formula('readable')
            
            results[method] = {
                'model': model,
                'report': report,
                'r2': metrics['train_r2'],
                'rmse': metrics['train_rmse'],
                'formula': formula
            }
            
            print(f"   R² = {metrics['train_r2']:.4f}")
            print(f"   RMSE = {metrics['train_rmse']:.2f} GPa")
            print(f"   公式: {formula[:80]}...")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            results[method] = {'error': str(e)}
    
    return results

def analyze_best_formula(results):
    """分析最佳公式"""
    print("\n🎯 最佳公式分析")
    print("=" * 60)
    
    # 找到最佳结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        print("❌ 没有有效的结果")
        return None
    
    best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['r2'])
    best_result = valid_results[best_method]
    
    print(f"🏆 最佳方法: {best_method}")
    print(f"   R² = {best_result['r2']:.4f}")
    print(f"   RMSE = {best_result['rmse']:.2f} GPa")
    print()
    
    # 显示完整报告
    print("📋 完整报告:")
    print(best_result['report'])
    
    return best_result

def physical_interpretation(best_result):
    """物理解释"""
    if not best_result:
        return
        
    print("\n🧮 物理解释")
    print("=" * 60)
    
    features = best_result['report']['results']['final_model']['features']
    
    print("发现的重要特征组合:")
    for i, feature in enumerate(features, 1):
        signature = feature['signature']
        coefficient = feature['coefficient']
        complexity = feature['complexity']
        
        print(f"{i}. {signature}")
        print(f"   系数: {coefficient:+.4f}")
        print(f"   复杂度: {complexity}")
        
        # 简单的物理解释
        if 'Electronegativity' in signature:
            print("   💡 电负性相关 - 影响化学键强度")
        if 'Atomic_Radius' in signature:
            print("   💡 原子半径相关 - 影响晶格参数和密度")
        if 'Valence_Electrons' in signature:
            print("   💡 价电子数相关 - 影响键合类型")
        if 'First_Ionization' in signature:
            print("   💡 电离能相关 - 反映元素活泼性")
        if 'square' in signature:
            print("   💡 平方项 - 可能反映非线性效应")
        print()

def main():
    """主函数"""
    print("🔬 材料科学SISSO应用：二元化合物体积模量预测")
    print("=" * 80)
    print("目标：从元素属性发现体积模量的显式解析公式")
    print()
    
    # 1. 加载数据
    print("📊 加载数据...")
    elements_df = load_element_properties()
    compounds_df = generate_binary_compounds_data()
    
    print(f"   元素属性数据: {len(elements_df)} 种元素")
    print(f"   化合物数据: {len(compounds_df)} 种二元化合物")
    print()
    
    # 2. 创建特征矩阵
    print("🔧 构建特征矩阵...")
    features_df = create_feature_matrix(compounds_df, elements_df)
    
    print(f"   特征维度: {features_df.shape}")
    print(f"   体积模量范围: {features_df['Bulk_Modulus'].min():.1f} - {features_df['Bulk_Modulus'].max():.1f} GPa")
    print()
    
    # 3. 相关性分析
    correlations = analyze_correlations(features_df)
    
    # 4. 准备SISSO输入
    X = features_df.drop('Bulk_Modulus', axis=1)
    y = features_df['Bulk_Modulus']
    
    print(f"\n📋 特征列表:")
    for i, col in enumerate(X.columns, 1):
        print(f"{i:2d}. {col}")
    
    # 5. SISSO分析
    results = run_sisso_analysis(X, y)
    
    # 6. 分析最佳公式
    best_result = analyze_best_formula(results)
    
    # 7. 物理解释
    physical_interpretation(best_result)
    
    print("\n🎉 分析完成！")
    print("💡 发现的公式可用于快速预测新材料的体积模量")
    print("🔬 建议：用更多实验数据验证和改进模型")

if __name__ == "__main__":
    main()
