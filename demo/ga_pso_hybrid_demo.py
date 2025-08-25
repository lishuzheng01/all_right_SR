# -*- coding: utf-8 -*-
"""
遗传算法+粒子群优化混合符号回归演示
==================================

结合遗传算法的全局搜索能力和粒子群优化的快速收敛特性，
实现更高效的符号回归求解。

特点：
- 双重优化策略
- 收敛速度快
- 避免局部最优
- 参数自适应调整
"""

import numpy as np
import pandas as pd
from sisso_py.evolutionary.ga_pso import GAPSORegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = x1*x2 + exp(x3/5) + noise"""
    np.random.seed(123)
    n_samples = 80
    
    # 生成特征数据
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-1.5, 1.5, n_samples)
    x3 = np.random.uniform(-3, 3, n_samples)
    
    # 真实函数关系
    y_true = x1 * x2 + np.exp(x3/5)
    y = y_true + np.random.normal(0, 0.05, n_samples)  # 添加噪声
    y = pd.Series(y, name='target')  # 转换为Series
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🔀 遗传算法+PSO混合符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = x1*x2 + exp(x3/5) + noise")
    
    # 创建GA-PSO混合模型
    print("\n🔧 创建GA-PSO混合模型...")
    ga_pso = GAPSORegressor(
        population_size=50,       # 种群大小
        generations=30,           # 进化代数
        crossover_prob=0.8,       # 交叉概率
        mutation_prob=0.15,       # 变异概率
        max_depth=6               # 表达式最大深度
    )
    
    # 训练模型
    print("🏃 开始训练...")
    ga_pso.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = ga_pso.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取最佳表达式
    print(f"\n🔍 GA-PSO混合算法找到了最佳表达式组合")
    
    # 算法优势分析
    print(f"\n📋 GA-PSO混合算法特点:")
    print(f"  🧬 遗传算法: 提供强大的全局搜索能力")
    print(f"  🚀 粒子群优化: 加速收敛到最优解")
    print(f"  🔄 混合策略: 平衡探索与开发")
    print(f"  ⚡ 收敛速度: 比单纯GA更快")
    
    # 性能评估
    if r2 > 0.8:
        print(f"\n✅ 模型性能优秀! (R² = {r2:.4f})")
    elif r2 > 0.6:
        print(f"\n👍 模型性能良好! (R² = {r2:.4f})")
    else:
        print(f"\n⚠️ 模型可能需要调优 (R² = {r2:.4f})")
        print(f"   建议: 增加种群大小或进化代数")
    
    print(f"\n💡 使用建议:")
    print(f"   - 对于复杂问题，可增加population_size和generations")
    print(f"   - 调整crossover_prob和mutation_prob平衡探索与开发")
    print(f"   - max_depth控制表达式复杂度，避免过拟合")

if __name__ == "__main__":
    main()
