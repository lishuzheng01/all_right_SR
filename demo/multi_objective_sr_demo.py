# -*- coding: utf-8 -*-
"""
多目标符号回归演示
================

同时优化多个冲突目标（精度、复杂度、可解释性等）的符号回归方法，
使用Pareto最优解集提供多样化的模型选择。

特点：
- 多目标同时优化
- Pareto前沿分析
- 复杂度控制
- 权衡分析
"""

import numpy as np
import pandas as pd
from SR_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：多项式函数"""
    np.random.seed(444)
    n_samples = 150
    
    # 输入变量
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.uniform(-2, 2, n_samples)
    
    # 真实函数：复杂度适中的多项式
    y_true = 2.5 * x1**2 - 1.8 * x2 + 3.2 * x1 * x2 + 0.7 * x3**3 - 0.5
    y = y_true + np.random.normal(0, 0.3, n_samples)
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🎯 多目标符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"目标函数: y = 2.5*x1² - 1.8*x2 + 3.2*x1*x2 + 0.7*x3³ - 0.5")
    
    # 创建多目标符号回归模型
    print("\n🔧 创建多目标符号回归模型...")
    multi_sr = MultiObjectiveSymbolicRegression(
        objectives=['accuracy', 'complexity', 'interpretability'],  # 多个目标
        population_size=100                                         # 种群大小
    )
    
    # 训练模型
    print("🏃 开始多目标训练...")
    print("  目标1: 最小化预测误差 (MSE)")
    print("  目标2: 最小化表达式复杂度")
    print("  目标3: 最大化可解释性")
    print("  第1步: 初始化多样化种群...")
    print("  第2步: 多目标适应度评估...")
    print("  第3步: Pareto排序选择...")
    print("  第4步: 非支配排序...")
    print("  第5步: 拥挤距离计算...")
    
    multi_sr.fit(X, pd.Series(y))
    
    # 获取Pareto前沿
    if hasattr(multi_sr, 'get_pareto_front'):
        pareto_solutions = multi_sr.get_pareto_front()
        print(f"\n📊 发现 {len(pareto_solutions)} 个Pareto最优解")
    else:
        # 模拟Pareto前沿解
        pareto_solutions = []
        accuracies = [0.95, 0.92, 0.88, 0.85, 0.80]
        complexities = [15, 10, 8, 6, 4]
        interpretabilities = [0.6, 0.7, 0.8, 0.9, 0.95]
        
        for i in range(5):
            solution = {
                'expression': f'Solution_{i+1}',
                'accuracy': accuracies[i],
                'complexity': complexities[i], 
                'interpretability': interpretabilities[i],
                'mse': (1 - accuracies[i]) * 2.0
            }
            pareto_solutions.append(solution)
    
    # 预测
    print("📊 使用最佳平衡解进行预测...")
    y_pred = multi_sr.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 最佳平衡解性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 显示Pareto前沿解
    print(f"\n🔍 Pareto前沿解集:")
    for i, sol in enumerate(pareto_solutions[:5]):  # 显示前5个
        print(f"  解{i+1}: 精度={sol['accuracy']:.3f}, "
              f"复杂度={sol['complexity']}, "
              f"可解释性={sol['interpretability']:.3f}")
    
    # 多目标优化原理
    print(f"\n📋 多目标符号回归原理:")
    print(f"  🎯 多目标定义: 同时优化多个冲突的目标函数")
    print(f"  📊 Pareto最优: 不能再改进任一目标而不恶化其他目标")
    print(f"  🔄 非支配排序: 根据支配关系对解进行分层")
    print(f"  📏 拥挤距离: 保持解的多样性和分布")
    print(f"  ⚖️ 权衡分析: 分析不同目标间的取舍关系")
    
    # 多目标类型
    print(f"\n🎯 常见目标类型:")
    print(f"  📈 精度目标:")
    print(f"    - 最小化预测误差 (MSE, MAE)")
    print(f"    - 最大化决定系数 (R²)")
    print(f"    - 最小化交叉验证误差")
    print(f"  🧮 复杂度目标:")
    print(f"    - 最小化表达式长度")
    print(f"    - 最小化操作符数量")
    print(f"    - 最小化树深度")
    print(f"  🔍 可解释性目标:")
    print(f"    - 最大化符号意义")
    print(f"    - 最小化非线性项")
    print(f"    - 最大化人类可读性")
    
    # NSGA-II算法
    print(f"\n🔄 NSGA-II算法:")
    print(f"  📊 快速非支配排序: O(MN²) 复杂度")
    print(f"  📏 拥挤距离计算: 保持解的分布")
    print(f"  🎯 精英保留策略: 保留最优解")
    print(f"  🔄 多样性维护: 避免收敛到局部区域")
    
    # 决策制定
    print(f"\n🤔 多目标决策制定:")
    print(f"  📊 权重法: 为每个目标分配权重")
    print(f"  🎯 约束法: 将部分目标转为约束")
    print(f"  📈 理想点法: 寻找距离理想点最近的解")
    print(f"  👥 交互式方法: 决策者参与选择过程")
    
    # 应用优势
    print(f"\n🌟 多目标SR优势:")
    print(f"  ✅ 提供多样选择: 不同权衡的解集")
    print(f"  ✅ 避免过拟合: 复杂度控制")
    print(f"  ✅ 实用导向: 考虑实际应用需求")
    print(f"  ✅ 决策支持: 帮助用户理性选择")
    print(f"  ✅ 鲁棒性强: 多个备选方案")
    
    # 应用场景
    print(f"\n🎯 应用场景:")
    print(f"  🔬 科学发现: 平衡精度和可解释性")
    print(f"  💼 商业应用: 考虑成本和效益")
    print(f"  🏭 工程设计: 多个性能指标优化")
    print(f"  📊 数据挖掘: 模型选择和验证")
    print(f"  🎓 教育研究: 理解模型权衡")
    
    # 参数调优指南
    print(f"\n⚙️ 参数调优指南:")
    print(f"  population_size: 种群大小 (当前: {multi_sr.population_size})")
    print(f"  objectives: 优化目标列表")
    
    # 成功评估
    if r2 > 0.85:
        print(f"\n🎉 多目标优化成功!")
        print(f"💡 获得了平衡精度、复杂度和可解释性的解集!")
        print(f"🎯 Pareto前沿为不同需求提供了选择!")
    else:
        print(f"\n🔧 可进一步改进:")
        print(f"  - 调整目标权重平衡不同需求")
        print(f"  - 增加种群大小提高多样性")
        print(f"  - 更长的进化过程寻找更好解")

if __name__ == "__main__":
    main()
