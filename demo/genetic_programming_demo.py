# -*- coding: utf-8 -*-
"""
遗传编程 (Genetic Programming) 符号回归演示
=============================================

遗传编程是一种基于进化算法的符号回归方法，通过模拟自然进化过程
自动发现数学表达式来描述数据中的隐藏规律。

特点：
- 自动搜索表达式结构
- 支持多种数学运算符
- 具有良好的全局搜索能力
- 可解释性强
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SR_py.evolutionary.gp import GeneticProgramming
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = 2*x1^2 + 0.5*sin(x2) + 0.1*x3 + noise"""
    np.random.seed(42)
    n_samples = 100
    
    # 生成特征数据
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-np.pi, np.pi, n_samples)
    x3 = np.random.uniform(-2, 2, n_samples)
    
    # 真实函数关系
    y_true = 2 * x1**2 + 0.5 * np.sin(x2) + 0.1 * x3
    y = y_true + np.random.normal(0, 0.1, n_samples)  # 添加噪声
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🧬 遗传编程符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = 2*x1² + 0.5*sin(x2) + 0.1*x3 + noise")
    
    # 创建遗传编程模型
    print("\n🔧 创建遗传编程模型...")
    gp = GeneticProgramming(
        population_size=100,      # 种群大小
        n_generations=50,         # 进化代数
        crossover_rate=0.8,       # 交叉率
        mutation_rate=0.2,        # 变异率
        max_depth=6,              # 树最大深度
        n_jobs=1                  # 并行进程数
    )
    
    # 训练模型
    print("🏃 开始训练...")
    gp.fit(X, y, feature_names=X.columns.tolist())
    
    # 预测
    print("📊 进行预测...")
    y_pred = gp.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取最佳表达式
    if hasattr(gp, 'get_best_model_string'):
        best_expr = gp.get_best_model_string()
        print(f"\n🔍 发现的最佳表达式:")
        print(f"y = {best_expr}")
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    
    # 真实值 vs 预测值
    plt.subplot(1, 3, 1)
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')
    plt.grid(True, alpha=0.3)
    
    # 残差图
    plt.subplot(1, 3, 2)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')
    plt.grid(True, alpha=0.3)
    
    # 时间序列预测对比
    plt.subplot(1, 3, 3)
    indices = np.arange(min(50, len(y)))
    plt.plot(indices, y[:len(indices)], 'b-', label='真实值', linewidth=2)
    plt.plot(indices, y_pred[:len(indices)], 'r--', label='预测值', linewidth=2)
    plt.xlabel('样本序号')
    plt.ylabel('目标值')
    plt.title('预测对比 (前50个样本)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出模型信息
    print(f"\n📋 模型参数:")
    print(f"  种群大小: {gp.population_size}")
    print(f"  进化代数: {gp.n_generations}")
    print(f"  交叉率: {gp.crossover_rate}")
    print(f"  变异率: {gp.mutation_rate}")
    print(f"  最大深度: {gp.max_depth}")
    
    print(f"\n✅ 遗传编程演示完成!")
    print(f"💡 提示: 遗传编程适合发现复杂的非线性关系，")
    print(f"    但可能需要较长的训练时间来收敛到最优解。")

if __name__ == "__main__":
    main()
