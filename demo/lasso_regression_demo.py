# -*- coding: utf-8 -*-
"""
LASSO稀疏回归符号回归演示
========================

LASSO (Least Absolute Shrinkage and Selection Operator) 通过L1正则化
实现特征选择和稀疏建模，适合发现简洁的线性和多项式关系。

特点：
- 自动特征选择
- 避免过拟合
- 模型简洁性
- 快速求解
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：多项式关系 y = 2*x1^2 + 1.5*x1*x2 - 0.8*x2^2 + 0.3*x3 + noise"""
    np.random.seed(789)
    n_samples = 100
    
    # 生成特征数据
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-1.5, 1.5, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    x4 = np.random.uniform(-0.5, 0.5, n_samples)  # 冗余特征
    
    # 真实函数关系 (多项式)
    y_true = 2 * x1**2 + 1.5 * x1 * x2 - 0.8 * x2**2 + 0.3 * x3
    y = y_true + np.random.normal(0, 0.1, n_samples)  # 添加噪声
    y = pd.Series(y, name='target')  # 转换为Series
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3,
        'x4': x4  # 冗余特征，应该被LASSO自动过滤
    })
    
    return X, y, y_true

def main():
    print("🎯 LASSO稀疏回归符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = 2*x1² + 1.5*x1*x2 - 0.8*x2² + 0.3*x3 + noise")
    print(f"注意: x4是冗余特征，应该被LASSO自动过滤")
    
    # 创建LASSO模型
    print("\n🔧 创建LASSO回归模型...")
    lasso = LassoRegressor(
        alpha=0.01,              # L1正则化强度
        max_iter=2000,           # 最大迭代次数
        poly_degree=2,           # 多项式特征扩展度
        normalize=True           # 特征标准化
    )
    
    # 训练模型
    print("🏃 开始训练...")
    print("  第1步: 构造多项式特征...")
    print("  第2步: LASSO稀疏回归...")
    print("  第3步: 交叉验证选择最优alpha...")
    
    lasso.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = lasso.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取模型信息
    if hasattr(lasso, 'get_model_info'):
        model_info = lasso.get_model_info()
        print(f"\n🔍 LASSO发现的稀疏公式:")
        
        if 'formula' in model_info:
            print(f"数学表达式: {model_info['formula']}")
        
        if 'selected_features' in model_info:
            print(f"选中特征数: {len(model_info['selected_features'])}")
            print(f"总特征数: {model_info.get('total_features', 'Unknown')}")
        
        if 'sparsity' in model_info:
            print(f"稀疏度: {model_info['sparsity']:.2%}")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 真实值 vs 预测值
    plt.subplot(2, 3, 1)
    plt.scatter(y, y_pred, alpha=0.7, c='blue', edgecolors='black', linewidth=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('真实值 vs 预测值')
    plt.grid(True, alpha=0.3)
    
    # 残差分析
    plt.subplot(2, 3, 2)
    residuals = y - y_pred
    plt.scatter(range(len(residuals)), residuals, alpha=0.7, c='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('样本序号')
    plt.ylabel('残差')
    plt.title('残差序列')
    plt.grid(True, alpha=0.3)
    
    # 系数路径 (不同alpha值下的系数变化)
    plt.subplot(2, 3, 3)
    alphas = np.logspace(-4, 1, 50)
    coeffs = []
    
    for alpha in alphas:
        temp_lasso = LassoRegressor(alpha=alpha, poly_degree=2, normalize=True)
        temp_lasso.fit(X, y)
        if hasattr(temp_lasso, 'get_model_info'):
            info = temp_lasso.get_model_info()
            coeffs.append(info.get('raw_coefficients', [0]))
        else:
            coeffs.append([0])
    
    if coeffs and len(coeffs[0]) > 1:
        coeffs = np.array(coeffs)
        for i in range(min(5, coeffs.shape[1])):  # 显示前5个系数
            plt.plot(alphas, coeffs[:, i], label=f'coef_{i+1}')
    
    plt.xscale('log')
    plt.xlabel('Alpha (正则化强度)')
    plt.ylabel('系数值')
    plt.title('LASSO正则化路径')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 特征重要性
    plt.subplot(2, 3, 4)
    feature_importance = [abs(np.corrcoef(X.iloc[:, i], y)[0, 1]) for i in range(X.shape[1])]
    bars = plt.bar(X.columns, feature_importance, alpha=0.7, 
                   color=['red', 'green', 'blue', 'orange'])
    plt.xlabel('原始特征')
    plt.ylabel('重要性 (|相关系数|)')
    plt.title('特征重要性分析')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 在柱子上添加数值
    for bar, importance in zip(bars, feature_importance):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{importance:.3f}', ha='center', va='bottom')
    
    # 真实函数 vs 预测函数在x1维度上的对比
    plt.subplot(2, 3, 5)
    x1_range = np.linspace(-2, 2, 50)
    x2_fixed = 0  # 固定x2=0
    x3_fixed = 0  # 固定x3=0
    x4_fixed = 0  # 固定x4=0
    
    X_test = pd.DataFrame({
        'x1': x1_range,
        'x2': [x2_fixed] * len(x1_range),
        'x3': [x3_fixed] * len(x1_range),
        'x4': [x4_fixed] * len(x1_range)
    })
    
    y_true_1d = 2 * x1_range**2  # 其他变量为0时的真实函数
    y_pred_1d = lasso.predict(X_test)
    
    plt.plot(x1_range, y_true_1d, 'b-', label='真实函数', linewidth=2)
    plt.plot(x1_range, y_pred_1d, 'r--', label='LASSO预测', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('y (x2=x3=x4=0)')
    plt.title('函数形状对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分布
    plt.subplot(2, 3, 6)
    plt.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频次')
    plt.title('残差分布')
    plt.axvline(x=0, color='red', linestyle='--', label='零线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # LASSO原理解释
    print(f"\n📋 LASSO稀疏回归原理:")
    print(f"  🎯 目标函数: MSE + α*||w||₁")
    print(f"  📉 L1正则化: 自动将不重要特征的系数压缩为0")
    print(f"  🔧 特征选择: 实现自动特征选择")
    print(f"  ⚖️  偏差-方差权衡: α控制模型复杂度")
    
    # 参数调优建议
    print(f"\n⚙️ 参数调优指南:")
    print(f"  alpha: 正则化强度 (当前: {lasso.alpha})")
    print(f"    - 过大: 欠拟合，所有系数趋于0")
    print(f"    - 过小: 过拟合，类似普通最小二乘")
    print(f"  poly_degree: 多项式度数 (当前: {lasso.poly_degree})")
    print(f"    - 增加可捕捉更复杂的非线性关系")
    print(f"  normalize: 特征标准化 (当前: {lasso.normalize})")
    print(f"    - 建议开启，确保正则化的公平性")
    
    # 应用场景
    print(f"\n🎯 LASSO适用场景:")
    print(f"  ✅ 线性或多项式关系")
    print(f"  ✅ 高维稀疏数据")
    print(f"  ✅ 需要特征选择")
    print(f"  ✅ 要求模型简洁可解释")
    print(f"  ❌ 强非线性关系可能需要其他方法")

if __name__ == "__main__":
    main()
