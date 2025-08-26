# -*- coding: utf-8 -*-
"""
概率程序归纳 (Probabilistic Program Induction) 演示
================================================

概率程序归纳使用概率上下文无关文法(PCFG)生成候选表达式，
通过贝叶斯更新不断优化文法概率来发现最优数学公式。

特点：
- 结构化程序生成
- 贝叶斯学习文法
- 组合性强
- 可扩展性好
"""

import numpy as np
import pandas as pd
from SR_py.probabilistic.ppi import ProbabilisticProgramInduction
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = x1 + sqrt(|x2|) + 0.2*x3^3 + noise"""
    np.random.seed(987)
    n_samples = 100
    
    # 生成特征数据
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    x3 = np.random.uniform(-1.5, 1.5, n_samples)
    
    # 真实函数关系
    y_true = x1 + np.sqrt(np.abs(x2)) + 0.2 * x3**3
    y = y_true + np.random.normal(0, 0.2, n_samples)  # 添加噪声
    y = pd.Series(y, name='target')  # 转换为Series
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🎯 概率程序归纳演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = x1 + √|x2| + 0.2*x3³ + noise")
    
    # 创建概率程序归纳模型
    print("\n🔧 创建概率程序归纳模型...")
    ppi = ProbabilisticProgramInduction(
        n_iterations=500,        # 搜索迭代次数
        population_size=100,     # 候选程序种群大小
        max_expr_depth=6,        # 表达式最大深度
        prior_temp=1.5           # 先验温度参数
    )
    
    # 训练模型
    print("🏃 开始程序归纳...")
    print("  第1步: 初始化PCFG...")
    print("  第2步: 生成候选程序...")
    print("  第3步: 评估程序性能...")
    print("  第4步: 贝叶斯更新文法...")
    print("  第5步: 迭代优化...")
    
    ppi.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = ppi.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取程序归纳结果
    if hasattr(ppi, 'get_model_info'):
        model_info = ppi.get_model_info()
        print(f"\n🔍 程序归纳结果:")
        
        if 'best_program' in model_info:
            print(f"最佳程序: {model_info['best_program']}")
            
        if 'program_probability' in model_info:
            print(f"程序概率: {model_info['program_probability']:.4f}")
            
        if 'grammar_entropy' in model_info:
            print(f"文法熵: {model_info['grammar_entropy']:.4f}")
        
        if 'convergence_iterations' in model_info:
            print(f"收敛迭代数: {model_info['convergence_iterations']}")
    
    # 概率程序归纳解释
    print(f"\n📋 概率程序归纳原理:")
    print(f"  📝 PCFG文法: 定义程序的生成规则")
    print(f"  🎲 随机采样: 按概率生成候选程序")
    print(f"  📊 贝叶斯更新: 根据数据调整文法概率")
    print(f"  🔄 迭代学习: 逐步提高程序质量")
    
    # 方法优势
    print(f"\n🌟 PPI方法优势:")
    print(f"  ✅ 结构化搜索: 系统性探索程序空间")
    print(f"  ✅ 组合性: 可发现复杂的组合结构")
    print(f"  ✅ 可解释性: 生成的程序易于理解")
    print(f"  ✅ 扩展性: 可轻松添加新的语法规则")
    print(f"  ✅ 先验知识: 可融入领域特定语法")
    
    # 与其他方法对比
    print(f"\n🔄 与其他方法对比:")
    print(f"  vs 遗传编程: 更系统化的搜索策略")
    print(f"  vs 神经网络: 更好的可解释性")
    print(f"  vs SISSO: 更灵活的表达式结构")
    print(f"  vs 贝叶斯SR: 更高效的采样方法")
    
    # 参数调优
    print(f"\n⚙️ 参数调优建议:")
    print(f"  n_iterations: 搜索迭代次数 (当前: {ppi.n_iterations})")
    print(f"  population_size: 种群大小 (当前: {ppi.population_size})")
    print(f"  max_expr_depth: 最大深度 (当前: {ppi.max_expr_depth})")
    print(f"  prior_temp: 先验温度 (当前: {ppi.prior_temp})")
    print(f"    - 控制探索vs开发的平衡")
    
    # 应用场景
    print(f"\n🎯 适用场景:")
    print(f"  🔬 科学计算: 发现物理、化学定律")
    print(f"  🤖 机器人学: 控制律的自动发现")
    print(f"  💰 金融工程: 复杂金融模型构建")
    print(f"  🧬 生物信息: 基因调控网络建模")
    print(f"  🌍 气候模型: 大气海洋耦合关系")
    
    # 成功评估
    if r2 > 0.8:
        print(f"\n🎉 程序归纳成功! 文法学习效果良好!")
        print(f"💡 PPI成功从数据中学习了有效的程序结构!")
    else:
        print(f"\n🔧 可进一步优化:")
        print(f"  - 增加迭代次数或种群大小")
        print(f"  - 调整文法规则或先验概率")
        print(f"  - 优化温度参数平衡探索与开发")

if __name__ == "__main__":
    main()
