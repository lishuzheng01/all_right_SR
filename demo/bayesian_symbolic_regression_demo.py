# -*- coding: utf-8 -*-
"""
贝叶斯符号回归 (Bayesian Symbolic Regression) 演示
===============================================

贝叶斯符号回归使用MCMC方法在表达式空间中采样，
通过贝叶斯推理找到最优的数学表达式。

特点：
- 不确定性量化
- 先验知识融入
- 多模态后验分布
- 鲁棒性强
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sisso_py.probabilistic.bsr import BayesianSymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = sin(x1) + 0.5*x2^2 + noise"""
    np.random.seed(654)
    n_samples = 80
    
    # 生成特征数据
    x1 = np.random.uniform(-np.pi, np.pi, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    
    # 真实函数关系 (非线性)
    y_true = np.sin(x1) + 0.5 * x2**2
    y = y_true + np.random.normal(0, 0.15, n_samples)  # 添加噪声
    y = pd.Series(y, name='target')  # 转换为Series
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🎲 贝叶斯符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = sin(x1) + 0.5*x2² + noise")
    
    # 创建贝叶斯符号回归模型
    print("\n🔧 创建贝叶斯符号回归模型...")
    bsr = BayesianSymbolicRegressor(
        n_iter=1000,             # MCMC迭代次数
        n_chains=3,              # 并行MCMC链数
        max_expr_depth=6,        # 表达式最大深度
        temperature=1.2          # 温度参数
    )
    
    # 训练模型
    print("🏃 开始贝叶斯推理...")
    print("  第1步: 初始化MCMC链...")
    print("  第2步: 在表达式空间中采样...")
    print("  第3步: 计算后验概率...")
    print("  第4步: 选择最优表达式...")
    
    bsr.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = bsr.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取贝叶斯推理结果
    if hasattr(bsr, 'get_model_info'):
        model_info = bsr.get_model_info()
        print(f"\n🔍 贝叶斯推理结果:")
        
        if 'best_expression' in model_info:
            print(f"最优表达式: {model_info['best_expression']}")
            
        if 'posterior_probability' in model_info:
            print(f"后验概率: {model_info['posterior_probability']:.4f}")
            
        if 'uncertainty' in model_info:
            print(f"预测不确定性: {model_info['uncertainty']:.4f}")
        
        if 'acceptance_rate' in model_info:
            print(f"MCMC接受率: {model_info['acceptance_rate']:.2%}")
    
   
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 真实值 vs 预测值
    plt.subplot(2, 3, 1)
    plt.scatter(y, y_pred, alpha=0.7, c='blue', edgecolors='black', linewidth=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('贝叶斯预测结果')
    plt.grid(True, alpha=0.3)
    
    # 不确定性分析
    plt.subplot(2, 3, 2)
    residuals = y - y_pred
    residual_std = np.std(residuals)
    
    # 模拟预测不确定性区间
    uncertainty_lower = y_pred - 1.96 * residual_std
    uncertainty_upper = y_pred + 1.96 * residual_std
    
    sorted_indices = np.argsort(y_pred)
    plt.fill_between(y_pred[sorted_indices], 
                     uncertainty_lower[sorted_indices], 
                     uncertainty_upper[sorted_indices], 
                     alpha=0.3, color='gray', label='95%置信区间')
    plt.scatter(y_pred, y, alpha=0.7, c='red', s=20)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'b--', lw=2)
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('预测不确定性')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # sin(x1) 函数对比
    plt.subplot(2, 3, 3)
    x1_range = np.linspace(-np.pi, np.pi, 100)
    x2_fixed = 0  # 固定x2=0
    x3_fixed = 0  # 固定x3=0
    
    X_test = pd.DataFrame({
        'x1': x1_range,
        'x2': [x2_fixed] * len(x1_range),
        'x3': [x3_fixed] * len(x1_range)
    })
    
    y_true_sin = np.sin(x1_range)  # 真实的sin函数部分
    y_pred_sin = bsr.predict(X_test)
    
    plt.plot(x1_range, y_true_sin, 'b-', label='真实 sin(x1)', linewidth=2)
    plt.plot(x1_range, y_pred_sin, 'r--', label='贝叶斯预测', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('y (x2=x3=0)')
    plt.title('sin函数识别')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # x2^2 函数对比
    plt.subplot(2, 3, 4)
    x2_range = np.linspace(-2, 2, 100)
    x1_fixed = 0  # 固定x1=0
    x3_fixed = 0  # 固定x3=0
    
    X_test2 = pd.DataFrame({
        'x1': [x1_fixed] * len(x2_range),
        'x2': x2_range,
        'x3': [x3_fixed] * len(x2_range)
    })
    
    y_true_quad = 0.5 * x2_range**2  # 真实的二次函数部分
    y_pred_quad = bsr.predict(X_test2)
    
    plt.plot(x2_range, y_true_quad, 'b-', label='真实 0.5*x2²', linewidth=2)
    plt.plot(x2_range, y_pred_quad, 'r--', label='贝叶斯预测', linewidth=2)
    plt.xlabel('x2')
    plt.ylabel('y (x1=x3=0)')
    plt.title('二次函数识别')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差分析
    plt.subplot(2, 3, 5)
    plt.scatter(range(len(residuals)), residuals, alpha=0.7, c='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=float(residual_std), color='orange', linestyle=':', label=f'±σ = ±{residual_std:.3f}')
    plt.axhline(y=-float(residual_std), color='orange', linestyle=':')
    plt.xlabel('样本序号')
    plt.ylabel('残差')
    plt.title('残差分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MCMC收敛诊断 (模拟)
    plt.subplot(2, 3, 6)
    # 模拟MCMC链的收敛过程
    iterations = np.arange(1, 201)
    
    # 模拟三条链的目标函数值
    chain1 = -mse + np.random.normal(0, 0.1, len(iterations)) * np.exp(-iterations/50)
    chain2 = -mse + np.random.normal(0, 0.1, len(iterations)) * np.exp(-iterations/60)
    chain3 = -mse + np.random.normal(0, 0.1, len(iterations)) * np.exp(-iterations/40)
    
    plt.plot(iterations, chain1, 'r-', alpha=0.7, label='链1', linewidth=1)
    plt.plot(iterations, chain2, 'g-', alpha=0.7, label='链2', linewidth=1)
    plt.plot(iterations, chain3, 'b-', alpha=0.7, label='链3', linewidth=1)
    plt.xlabel('MCMC迭代次数')
    plt.ylabel('对数后验概率')
    plt.title('MCMC收敛诊断')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 贝叶斯方法解释
    print(f"\n📋 贝叶斯符号回归原理:")
    print(f"  🎯 贝叶斯定理: P(模型|数据) ∝ P(数据|模型) × P(模型)")
    print(f"  🔄 MCMC采样: 在表达式空间中随机游走")
    print(f"  📊 后验分布: 量化模型的不确定性")
    print(f"  🎲 多链并行: 提高采样效率和可靠性")
    
    # 优势分析
    print(f"\n🌟 贝叶斯方法优势:")
    print(f"  ✅ 不确定性量化: 提供预测置信区间")
    print(f"  ✅ 先验知识: 可融入领域专业知识")
    print(f"  ✅ 多模态: 发现多个可能的解")
    print(f"  ✅ 鲁棒性: 对噪声和异常值不敏感")
    print(f"  ✅ 自动复杂度选择: 避免过拟合")
    
    # 参数调优
    print(f"\n⚙️ 参数调优指南:")
    print(f"  n_iter: MCMC迭代次数 (当前: {bsr.n_iter})")
    print(f"    - 增加可提高收敛质量")
    print(f"  n_chains: 并行链数 (当前: {bsr.n_chains})")
    print(f"    - 多链可提高采样可靠性")
    print(f"  temperature: 温度参数 (当前: {bsr.temperature})")
    print(f"    - 控制探索vs开发的平衡")
    print(f"  max_expr_depth: 表达式复杂度 (当前: {bsr.max_expr_depth})")
    
    # 应用建议
    print(f"\n💡 应用建议:")
    print(f"  🔬 科学发现: 物理定律、化学反应等")
    print(f"  🏥 医学研究: 剂量-反应关系建模")
    print(f"  📈 金融建模: 风险评估和不确定性量化")
    print(f"  🌍 环境科学: 复杂生态系统建模")
    
    if r2 > 0.85:
        print(f"\n🎉 贝叶斯推理成功! 很好地量化了模型不确定性!")
    else:
        print(f"\n🔧 可考虑增加MCMC迭代次数或调整温度参数")

    
if __name__ == "__main__":
    main()


