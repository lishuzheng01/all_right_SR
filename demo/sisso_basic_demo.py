# -*- coding: utf-8 -*-
"""
SISSO (Sure Independence Screening and Sparsifying Operator) 演示
==============================================================

SISSO是一种基于特征构造和稀疏回归的符号回归方法，
通过系统地构造特征空间并筛选最相关的特征来发现数学公式。

特点：
- 系统化的特征构造
- 高效的特征筛选
- 稀疏化建模
- 可控的复杂度
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SR_py.sparse_regression.sisso import SISSORegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = 3*x1^2 + log(|x2|+0.1) - 0.5*x3 + noise"""
    np.random.seed(456)
    n_samples = 120
    
    # 生成特征数据
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    x4 = np.random.uniform(0.5, 2.5, n_samples)
    
    # 真实函数关系
    y_true = 3 * x1**2 + np.log(np.abs(x2) + 0.1) - 0.5 * x3
    y = y_true + np.random.normal(0, 0.1, n_samples)  # 添加噪声
    y = pd.Series(y, name='target')  # 转换为Series
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3,
        'x4': x4
    })
    
    return X, y, y_true

def main():
    print("🎯 SISSO符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = 3*x1² + log(|x2|+0.1) - 0.5*x3 + noise")
    
    # 创建SISSO模型
    print("\n🔧 创建SISSO模型...")
    sisso = SISSORegressor(
        K=3,                      # 复杂度层级
        sis_topk=100,            # SIS筛选保留的特征数
        so_max_terms=5,          # SO稀疏化最大项数
        sis_screener='pearson',   # 筛选方法
        so_solver='lasso',        # 稀疏求解器
        cv=5                      # 交叉验证折数
    )
    
    # 训练模型
    print("🏃 开始训练...")
    print("  第1步: 构造特征空间...")
    print("  第2步: SIS筛选重要特征...")  
    print("  第3步: SO稀疏化建模...")
    
    sisso.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = sisso.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取模型信息
    if hasattr(sisso, 'get_model_info'):
        model_info = sisso.get_model_info()
        print(f"\n🔍 SISSO发现的最佳表达式:")
        if 'selected_features' in model_info:
            features = model_info['selected_features']
            coefficients = model_info.get('coefficients', [])
            
            print("选中的特征及其系数:")
            for i, (feature, coef) in enumerate(zip(features, coefficients)):
                print(f"  {coef:8.4f} * {feature}")
            
            if 'intercept' in model_info:
                print(f"  {model_info['intercept']:8.4f} (截距)")
    
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
    
    # 残差图
    plt.subplot(2, 3, 2)
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7, c='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')
    plt.grid(True, alpha=0.3)
    
    # 误差分布
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频次')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)
    
    # 原始特征相关性
    plt.subplot(2, 3, 4)
    correlations = [np.corrcoef(X.iloc[:, i], y)[0, 1] for i in range(X.shape[1])]
    bars = plt.bar(X.columns, correlations, alpha=0.7, 
                   color=['red' if c > 0 else 'blue' for c in correlations])
    plt.xlabel('原始特征')
    plt.ylabel('与目标的相关性')
    plt.title('原始特征重要性')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 预测序列对比
    plt.subplot(2, 3, 5)
    indices = np.arange(min(50, len(y)))
    plt.plot(indices, y[:len(indices)], 'b-', label='真实值', linewidth=2)
    plt.plot(indices, y_pred[:len(indices)], 'r--', label='预测值', linewidth=2)
    plt.xlabel('样本序号')
    plt.ylabel('目标值')
    plt.title('预测序列对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SISSO算法流程示意
    plt.subplot(2, 3, 6)
    steps = ['原始特征', 'K层特征构造', 'SIS筛选', 'SO稀疏化', '最终模型']
    y_pos = np.arange(len(steps))
    plt.barh(y_pos, [4, 100, 100, 5, 1], alpha=0.7, 
             color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'])
    plt.yticks(y_pos, steps)
    plt.xlabel('特征数量 (示意)')
    plt.title('SISSO算法流程')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 算法解释
    print(f"\n📋 SISSO算法详解:")
    print(f"  🏗️  特征构造: 通过数学运算组合原始特征")
    print(f"  🔍 SIS筛选: 使用相关性筛选最重要的特征")
    print(f"  🎯 SO稀疏化: 使用LASSO等方法选择最终特征")
    print(f"  ⚖️  平衡: 在精度和简洁性之间取得平衡")
    
    # 参数说明
    print(f"\n⚙️ 关键参数说明:")
    print(f"  K: 特征构造的复杂度层级 (当前: {sisso.K})")
    print(f"  sis_topk: SIS筛选保留的特征数 (当前: {sisso.sis_topk})")
    print(f"  so_max_terms: 最终模型的最大项数 (当前: {sisso.so_max_terms})")
    print(f"  sis_screener: 特征筛选方法 (当前: {sisso.sis_screener})")
    print(f"  so_solver: 稀疏求解器 (当前: {sisso.so_solver})")
    
    # 性能建议
    if r2 > 0.95:
        print(f"\n🌟 优秀! SISSO成功发现了数据的潜在规律!")
    elif r2 > 0.8:
        print(f"\n👍 良好! 可考虑增加K值或调整筛选参数进一步优化")
    else:
        print(f"\n💡 建议调优:")
        print(f"   - 增加K值以构造更复杂的特征")
        print(f"   - 调整sis_topk保留更多候选特征")
        print(f"   - 尝试不同的筛选器和求解器")

if __name__ == "__main__":
    main()
