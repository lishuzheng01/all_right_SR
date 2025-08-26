# -*- coding: utf-8 -*-
"""
强化学习符号回归演示
==================

使用强化学习智能体在表达式空间中学习最优的数学公式，
通过奖励机制引导智能体发现高质量的符号表达式。

特点：
- 智能决策制定
- 序列化构建表达式
- 自适应探索策略
- 端到端学习
"""

import numpy as np
import pandas as pd
from SR_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = x1^2 - 2*x1*x2 + x2^2 + noise (完全平方式)"""
    np.random.seed(111)
    n_samples = 90
    
    # 生成特征数据
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)  # 干扰特征
    
    # 真实函数关系 (完全平方式)
    y_true = (x1 - x2)**2  # 等价于 x1^2 - 2*x1*x2 + x2^2
    y = y_true + np.random.normal(0, 0.1, n_samples)  # 添加噪声
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🤖 强化学习符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = (x1-x2)² = x1² - 2*x1*x2 + x2² + noise")
    
    # 创建强化学习符号回归模型
    print("\n🔧 创建强化学习模型...")
    rl_sr = ReinforcementSymbolicRegression(
        agent_type='dqn',        # 深度Q网络智能体
        max_episodes=200,        # 最大训练轮数
        batch_size=32,           # 批大小
        learning_rate=0.001      # 学习率
    )
    
    # 训练模型
    print("🏃 开始强化学习训练...")
    print("  第1步: 初始化Q网络...")
    print("  第2步: 探索表达式空间...")
    print("  第3步: 计算奖励信号...")
    print("  第4步: 更新智能体策略...")
    print("  第5步: 收敛到最优策略...")
    
    rl_sr.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = rl_sr.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取强化学习结果
    if hasattr(rl_sr, 'get_model_info'):
        model_info = rl_sr.get_model_info()
        print(f"\n🔍 强化学习结果:")
        
        if 'best_expression' in model_info:
            print(f"发现的表达式: {model_info['best_expression']}")
            
        if 'total_reward' in model_info:
            print(f"累积奖励: {model_info['total_reward']:.4f}")
            
        if 'exploration_rate' in model_info:
            print(f"最终探索率: {model_info['exploration_rate']:.4f}")
        
        if 'convergence_episode' in model_info:
            print(f"收敛轮数: {model_info['convergence_episode']}")
    
    # 强化学习原理解释
    print(f"\n📋 强化学习符号回归原理:")
    print(f"  🎯 马尔可夫决策过程: 状态→动作→奖励→新状态")
    print(f"  🧠 深度Q网络: 学习状态-动作价值函数")
    print(f"  🎲 ε-贪婪策略: 平衡探索与开发")
    print(f"  🔄 经验回放: 提高样本利用效率")
    print(f"  🎁 奖励设计: 引导智能体发现好的表达式")
    
    # 奖励函数设计
    print(f"\n🎁 奖励函数设计:")
    print(f"  📈 精度奖励: 基于预测精度的正奖励")
    print(f"  📉 复杂度惩罚: 过复杂表达式的负奖励")
    print(f"  ⭐ 完成奖励: 成功构建表达式的额外奖励")
    print(f"  💀 无效惩罚: 无效操作的负奖励")
    print(f"  🏆 发现奖励: 找到新颖结构的奖励")
    
    # 智能体架构
    print(f"\n🏗️ 智能体架构:")
    print(f"  🔍 状态表示: 当前表达式状态编码")
    print(f"  ⚡ 动作空间: 可执行的构建操作")
    print(f"  🧮 神经网络: DQN估计动作价值")
    print(f"  💾 经验池: 存储和重放历史经验")
    print(f"  🎛️ 目标网络: 稳定Q学习过程")
    
    # 优势与挑战
    print(f"\n🌟 RL-SR优势:")
    print(f"  ✅ 自适应搜索: 智能体自主学习搜索策略")
    print(f"  ✅ 序列决策: 适合表达式的序列构建")
    print(f"  ✅ 端到端: 直接优化最终目标")
    print(f"  ✅ 可扩展: 容易添加新的操作和约束")
    
    print(f"\n⚠️ 面临挑战:")
    print(f"  🎯 奖励设计: 需要精心设计奖励函数")
    print(f"  🔄 训练时间: 需要大量的探索和学习")
    print(f"  📊 样本效率: 可能需要很多样本才能收敛")
    print(f"  🎲 随机性: 结果可能存在一定随机性")
    
    # 参数调优
    print(f"\n⚙️ 参数调优指南:")
    print(f"  max_episodes: 训练轮数 (当前: {rl_sr.max_episodes})")
    print(f"  learning_rate: 学习率 (当前: {rl_sr.learning_rate})")
    print(f"  batch_size: 批大小 (当前: {rl_sr.batch_size})")
    print(f"  agent_type: 智能体类型 (当前: {rl_sr.agent_type})")
    
    # 改进方向
    print(f"\n🚀 改进方向:")
    print(f"  🔧 层次化RL: 分层决策，先选择结构再选择参数")
    print(f"  🤝 多智能体: 协作搜索不同部分的表达式")
    print(f"  🎭 元学习: 快速适应新的符号回归任务")
    print(f"  🌐 图神经网络: 更好地表示表达式结构")
    
    # 性能评估
    if r2 > 0.7:
        print(f"\n🎉 强化学习成功学会了构建表达式!")
        print(f"💡 智能体找到了有效的符号组合策略!")
    else:
        print(f"\n🔧 需要进一步训练:")
        print(f"  - 增加训练轮数")
        print(f"  - 调整奖励函数")
        print(f"  - 优化网络架构")

if __name__ == "__main__":
    main()
