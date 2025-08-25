# -*- coding: utf-8 -*-
"""
SINDy (Sparse Identification of Nonlinear Dynamics) 演示
=======================================================

SINDy专门用于发现动力学系统的控制方程，
通过稀疏回归识别系统的非线性动力学特征。

特点：
- 动力学系统建模
- 稀疏系数识别
- 物理解释性强
- 支持多种基函数
"""

import numpy as np
import pandas as pd
from sisso_py.sparse_regression.sindy import SINDyRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成动力学演示数据：模拟简化的洛伦兹系统"""
    np.random.seed(321)
    n_samples = 150
    dt = 0.02
    
    # 初始条件
    x0, y0, z0 = 1.0, 1.0, 1.0
    
    # 洛伦兹参数 (简化版)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    # 数值积分生成时间序列
    t = np.arange(0, n_samples * dt, dt)
    x, y, z = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    
    x[0], y[0], z[0] = x0, y0, z0
    
    for i in range(1, n_samples):
        # 简化的洛伦兹方程
        dxdt = sigma * (y[i-1] - x[i-1])
        dydt = x[i-1] * (rho - z[i-1]) - y[i-1]
        dzdt = x[i-1] * y[i-1] - beta * z[i-1]
        
        x[i] = x[i-1] + dxdt * dt
        y[i] = y[i-1] + dydt * dt  
        z[i] = z[i-1] + dzdt * dt
    
    # 构造特征矩阵 (当前状态)
    X = pd.DataFrame({
        'x': x[:-1],  # 当前时刻的x
        'y': y[:-1],  # 当前时刻的y
        'z': z[:-1]   # 当前时刻的z
    })
    
    # 目标是预测x的导数
    dxdt_true = sigma * (y[:-1] - x[:-1])
    target = pd.Series(dxdt_true + np.random.normal(0, 0.1, len(dxdt_true)), name='dxdt')
    
    return X, target, dxdt_true, t[:-1]

def main():
    print("🌀 SINDy动力学系统识别演示")
    print("=" * 50)
    
    # 生成演示数据
    X, target, dxdt_true, t = generate_demo_data()
    print(f"数据集大小: {len(X)} 时间步, {X.shape[1]} 状态变量")
    print(f"目标: 识别 dx/dt = σ(y-x) 的动力学方程")
    print(f"真实参数: σ = 10.0")
    
    # 创建SINDy模型
    print("\n🔧 创建SINDy模型...")
    sindy = SINDyRegressor(
        threshold=0.1,           # 稀疏阈值
        alpha=0.01,             # 正则化参数
        poly_degree=2,          # 多项式基函数度数
        solver='lasso'          # 稀疏求解器
    )
    
    # 训练模型
    print("🏃 开始训练...")
    print("  第1步: 构造多项式特征库...")
    print("  第2步: 稀疏回归识别活跃项...")
    print("  第3步: 阈值化获得最终方程...")
    
    sindy.fit(X, target)
    
    # 预测
    print("📊 进行预测...")
    y_pred = sindy.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(target, y_pred)
    r2 = r2_score(target, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取发现的方程
    if hasattr(sindy, 'get_model_info'):
        model_info = sindy.get_model_info()
        print(f"\n🔍 SINDy发现的动力学方程:")
        
        if 'formula' in model_info:
            print(f"dx/dt = {model_info['formula']}")
            
        if 'selected_features' in model_info:
            features = model_info['selected_features']
            coefficients = model_info.get('coefficients', [])
            print(f"\n活跃项分析:")
            for feature, coef in zip(features, coefficients):
                print(f"  {coef:8.4f} * {feature}")
        
        if 'sparsity' in model_info:
            print(f"\n稀疏度: {model_info['sparsity']:.2%}")
    
    # SINDy方法解释
    print(f"\n📋 SINDy方法解析:")
    print(f"  📚 特征库构造: 多项式、三角函数等基函数")
    print(f"  🎯 稀疏回归: 识别活跃的动力学项")
    print(f"  ✂️  阈值化: 去除小系数项获得简洁方程")
    print(f"  🔄 迭代优化: 提高稀疏性和精度")
    
    # 应用领域
    print(f"\n🌟 SINDy应用领域:")
    print(f"  🌪️  流体力学: 湍流、对流等")
    print(f"  🧬 生物系统: 种群动力学、生化反应")
    print(f"  ⚡ 工程控制: 机器人、航空航天")
    print(f"  🌍 气候科学: 大气海洋动力学")
    print(f"  💊 药物动力学: 药物代谢建模")
    
    # 参数调优建议
    print(f"\n⚙️ 参数调优建议:")
    print(f"  threshold: 稀疏阈值 (当前: {sindy.threshold})")
    print(f"    - 过大: 可能丢失重要项")
    print(f"    - 过小: 保留过多噪声项")
    print(f"  alpha: 正则化强度 (当前: {sindy.alpha})")
    print(f"  poly_degree: 多项式度数 (当前: {sindy.poly_degree})")
    print(f"    - 增加可捕捉更复杂的非线性")
    
    # 成功标准
    if r2 > 0.9:
        print(f"\n🎉 成功识别动力学方程! (R² = {r2:.4f})")
        print(f"💡 SINDy成功从数据中发现了潜在的物理规律!")
    else:
        print(f"\n🔧 可进一步优化 (R² = {r2:.4f})")
        print(f"💡 建议调整threshold或增加数据量")

if __name__ == "__main__":
    main()
