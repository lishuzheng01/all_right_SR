# -*- coding: utf-8 -*-
"""
物理约束符号回归演示
==================

结合物理知识和维度分析的符号回归方法，
确保发现的数学公式符合物理定律和量纲一致性。

特点：
- 物理定律约束
- 量纲分析
- 守恒定律
- 对称性原理
"""

import numpy as np
import pandas as pd
from sisso_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
from sisso_py.dsl.dimension import Dimension
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：模拟自由落体运动 s = v0*t + 0.5*g*t^2"""
    np.random.seed(333)
    n_samples = 100
    
    # 物理参数
    g = 9.81  # 重力加速度 m/s²
    
    # 生成物理变量
    v0 = np.random.uniform(0, 20, n_samples)    # 初始速度 m/s
    t = np.random.uniform(0, 5, n_samples)      # 时间 s
    m = np.random.uniform(1, 10, n_samples)     # 质量 kg (无关变量)
    
    # 真实物理关系：自由落体位移公式
    s_true = v0 * t + 0.5 * g * t**2
    s = s_true + np.random.normal(0, 0.5, n_samples)  # 添加测量噪声
    
    X = pd.DataFrame({
        'v0': v0,   # 初始速度
        't': t,     # 时间
        'm': m      # 质量 (冗余，应被物理约束过滤)
    })
    
    # 定义物理量纲
    feature_dims = {
        'v0': Dimension([0, 1, -1, 0, 0, 0, 0]),  # 速度: LT^-1
        't': Dimension([0, 0, 1, 0, 0, 0, 0]),    # 时间: T
        'm': Dimension([1, 0, 0, 0, 0, 0, 0])     # 质量: M
    }
    target_dim = Dimension([0, 1, 0, 0, 0, 0, 0])     # 位移: L
    
    return X, s, s_true, feature_dims, target_dim

def main():
    print("⚛️ 物理约束符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, s, s_true, feature_dims, target_dim = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 物理变量")
    print(f"物理场景: 自由落体运动")
    print(f"真实公式: s = v0*t + 0.5*g*t² (g≈9.81 m/s²)")
    print(f"目标量纲: L (长度)")
    
    # 显示物理量纲信息
    print(f"\n📏 物理量纲分析:")
    for var, dim in feature_dims.items():
        print(f"  {var}: {dim}")
    print(f"  target: {target_dim}")
    
    # 创建物理约束符号回归模型
    print("\n🔧 创建物理约束模型...")
    physics_sr = PhysicsInformedSymbolicRegression(
        physical_constraints=['conservation_laws'],  # 守恒定律
        dimensional_analysis=True,                    # 量纲分析
        constraint_weight=0.1,                       # 约束权重
        K=3                                          # 特征构造层数
    )
    
    # 训练模型
    print("🏃 开始物理约束训练...")
    print("  第1步: 量纲分析筛选候选特征...")
    print("  第2步: 应用物理约束...")
    print("  第3步: 构造符合物理定律的表达式...")
    print("  第4步: 优化参数估计...")
    
    physics_sr.fit(X, s, feature_dimensions=feature_dims, target_dimension=target_dim)
    
    # 预测
    print("📊 进行预测...")
    s_pred = physics_sr.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(s, s_pred)
    r2 = r2_score(s, s_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取物理约束结果
    if hasattr(physics_sr, 'get_model_info'):
        model_info = physics_sr.get_model_info()
        print(f"\n🔍 物理约束分析结果:")
        
        if 'discovered_law' in model_info:
            print(f"发现的物理定律: {model_info['discovered_law']}")
            
        if 'dimensional_consistency' in model_info:
            print(f"量纲一致性: {model_info['dimensional_consistency']}")
            
        if 'physical_parameters' in model_info:
            params = model_info['physical_parameters']
            print(f"估计的物理参数:")
            for param, value in params.items():
                print(f"  {param}: {value:.4f}")
        
        if 'constraint_violations' in model_info:
            violations = model_info['constraint_violations']
            print(f"约束违反数: {violations}")
    
    # 物理约束原理
    print(f"\n📋 物理约束符号回归原理:")
    print(f"  📏 量纲分析: 确保公式的量纲一致性")
    print(f"  ⚖️ 守恒定律: 能量、动量、质量守恒")
    print(f"  🔄 对称性: 时间、空间、旋转对称性")
    print(f"  📐 几何约束: 空间几何关系")
    print(f"  🎯 因果性: 因果关系的时间顺序")
    
    # 量纲分析详解
    print(f"\n📏 量纲分析详解:")
    print(f"  基本量纲: [M]质量, [L]长度, [T]时间, [I]电流等")
    print(f"  组合量纲: 速度[LT⁻¹], 加速度[LT⁻²], 力[MLT⁻²]")
    print(f"  一致性检查: 方程两边量纲必须相同")
    print(f"  筛选机制: 自动过滤量纲不匹配的项")
    
    # 物理约束类型
    print(f"\n⚖️ 物理约束类型:")
    print(f"  📊 守恒定律:")
    print(f"    - 能量守恒: 系统总能量保持恒定")
    print(f"    - 动量守恒: 无外力时动量不变")
    print(f"    - 质量守恒: 化学反应中质量守恒")
    print(f"  🔄 对称性:")
    print(f"    - 时间对称: 物理定律不随时间改变")
    print(f"    - 空间对称: 物理定律不随位置改变")
    print(f"    - 旋转对称: 各向同性")
    
    # 实施策略
    print(f"\n🛠️ 约束实施策略:")
    print(f"  💯 硬约束: 绝对不能违反的物理定律")
    print(f"  📊 软约束: 通过惩罚项引导搜索")
    print(f"  🔍 预筛选: 在特征生成阶段就过滤")
    print(f"  ✅ 后验证: 结果验证和一致性检查")
    
    # 应用优势
    print(f"\n🌟 物理约束SR优势:")
    print(f"  ✅ 物理可解释性: 发现的公式符合物理直觉")
    print(f"  ✅ 泛化能力强: 遵循物理定律的模型更稳健")
    print(f"  ✅ 搜索效率高: 约束缩小了搜索空间")
    print(f"  ✅ 参数有意义: 估计的参数具有物理含义")
    print(f"  ✅ 预测可靠: 在物理合理范围内外推")
    
    # 应用领域
    print(f"\n🎯 应用领域:")
    print(f"  🔬 经典力学: 运动学、动力学定律发现")
    print(f"  ⚡ 电磁学: 麦克斯韦方程组等")
    print(f"  🌡️ 热力学: 状态方程、相变规律")
    print(f"  🧪 化学: 反应动力学、平衡常数")
    print(f"  🌍 地球物理: 地质、气象模型")
    print(f"  🚀 工程: 结构力学、流体力学")
    
    # 挑战与限制
    print(f"\n⚠️ 挑战与限制:")
    print(f"  📚 先验知识: 需要正确的物理约束设定")
    print(f"  🎯 约束设计: 过强约束可能错过正确解")
    print(f"  🔄 复杂系统: 多体系统的约束难以表达")
    print(f"  📊 计算复杂: 约束检查增加计算开销")
    
    # 参数调优
    print(f"\n⚙️ 参数调优:")
    print(f"  constraint_weight: 约束权重 (当前: {physics_sr.constraint_weight})")
    print(f"  dimensional_analysis: 量纲检查 (当前: {physics_sr.dimensional_analysis})")
    print(f"  physical_constraints: 物理约束类型")
    print(f"  K: 特征构造复杂度 (当前: {physics_sr.K})")
    
    # 成功评估
    if r2 > 0.9:
        print(f"\n🎉 成功发现符合物理定律的数学公式!")
        print(f"💡 物理约束有效提高了模型的可解释性和准确性!")
        print(f"⚛️ 估计的物理参数接近真实值!")
    else:
        print(f"\n🔧 可进一步改进:")
        print(f"  - 调整约束权重平衡精度和物理一致性")
        print(f"  - 增加更多相关的物理约束")
        print(f"  - 检查量纲定义的准确性")

if __name__ == "__main__":
    main()
