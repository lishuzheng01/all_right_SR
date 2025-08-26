# -*- coding: utf-8 -*-
"""
深度符号回归演示
================

使用深度神经网络学习数据到符号表达式的映射，
通过编码器-解码器架构实现端到端的符号发现。

特点：
- 端到端学习
- 表达式编码解码
- 注意力机制
- 大规模并行训练
"""

import numpy as np
import pandas as pd
from SR_py.neural_symbolic.deep_sr import DeepSymbolicRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：y = exp(-x1^2) * cos(x2) + noise (高斯-余弦复合函数)"""
    np.random.seed(222)
    n_samples = 120
    
    # 生成特征数据
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-np.pi, np.pi, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)  # 冗余特征
    
    # 真实函数关系 (复杂非线性)
    y_true = np.exp(-x1**2) * np.cos(x2)
    y = y_true + np.random.normal(0, 0.05, n_samples)  # 添加噪声
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🧠 深度符号回归演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"真实函数: y = exp(-x1²) * cos(x2) + noise")
    
    # 创建深度符号回归模型
    print("\n🔧 创建深度符号回归模型...")
    deep_sr = DeepSymbolicRegression(
        encoder_layers=[64, 32],    # 编码器层配置
        decoder_layers=[32, 64],    # 解码器层配置
        max_length=20,              # 最大表达式长度
        epochs=100                  # 训练轮数
    )
    
    # 训练模型
    print("🏃 开始深度学习训练...")
    print("  第1步: 构建编码器-解码器网络...")
    print("  第2步: 数据特征编码...")
    print("  第3步: 表达式序列解码...")
    print("  第4步: 端到端反向传播...")
    print("  第5步: 模型收敛...")
    
    deep_sr.fit(X, y)
    
    # 预测
    print("📊 进行预测...")
    y_pred = deep_sr.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 模型性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取深度学习结果
    if hasattr(deep_sr, 'get_model_info'):
        model_info = deep_sr.get_model_info()
        print(f"\n🔍 深度学习结果:")
        
        if 'discovered_expression' in model_info:
            print(f"发现的表达式: {model_info['discovered_expression']}")
            
        if 'attention_weights' in model_info:
            print(f"注意力权重分布: 可视化特征重要性")
            
        if 'training_loss' in model_info:
            print(f"最终训练损失: {model_info['training_loss']:.4f}")
        
        if 'expression_complexity' in model_info:
            print(f"表达式复杂度: {model_info['expression_complexity']}")
    
    # 深度符号回归原理
    print(f"\n📋 深度符号回归原理:")
    print(f"  🔗 编码器-解码器: 数据→潜在表示→表达式")
    print(f"  📝 序列到序列: 将数值映射为符号序列")
    print(f"  👁️ 注意力机制: 关注重要的特征和操作")
    print(f"  🎯 端到端优化: 直接优化最终表达式质量")
    print(f"  🔄 反向传播: 梯度驱动的参数学习")
    
    # 网络架构解析
    print(f"\n🏗️ 网络架构解析:")
    print(f"  📥 编码器: 将数据特征编码为潜在表示")
    print(f"    - 多层感知机提取非线性特征")
    print(f"    - 批标准化加速收敛")
    print(f"    - Dropout防止过拟合")
    print(f"  📤 解码器: 将潜在表示解码为表达式")
    print(f"    - 递归神经网络生成符号序列")
    print(f"    - 注意力机制选择重要信息")
    print(f"    - Beam搜索优化序列生成")
    
    # 训练策略
    print(f"\n🎯 训练策略:")
    print(f"  📊 数据增强: 生成大量合成表达式")
    print(f"  🔧 课程学习: 从简单到复杂逐步训练")
    print(f"  ⚖️ 多任务学习: 同时学习多种数学关系")
    print(f"  🎲 对抗训练: 提高模型鲁棒性")
    print(f"  🔄 迁移学习: 预训练模型快速适应")
    
    # 优势与限制
    print(f"\n🌟 深度SR优势:")
    print(f"  ✅ 大规模学习: 能处理大量数据和复杂模式")
    print(f"  ✅ 端到端优化: 直接优化最终目标")
    print(f"  ✅ 表示学习: 自动学习有用的特征表示")
    print(f"  ✅ 并行计算: 充分利用GPU加速")
    print(f"  ✅ 泛化能力: 可迁移到相似问题")
    
    print(f"\n⚠️ 当前限制:")
    print(f"  📊 数据需求: 需要大量训练数据")
    print(f"  ⚫ 黑盒性质: 决策过程不够透明")
    print(f"  🎯 表达式质量: 可能生成复杂但不简洁的表达式")
    print(f"  💻 计算资源: 需要大量计算资源")
    print(f"  🎲 随机性: 训练结果可能不稳定")
    
    # 参数调优
    print(f"\n⚙️ 参数调优指南:")
    print(f"  encoder_layers: 编码器层配置 (当前: {deep_sr.encoder_layers})")
    print(f"  decoder_layers: 解码器层配置 (当前: {deep_sr.decoder_layers})")
    print(f"  max_length: 最大表达式长度 (当前: {deep_sr.max_length})")
    print(f"  epochs: 训练轮数 (当前: {deep_sr.epochs})")
    
    # 前沿发展
    print(f"\n🚀 前沿发展方向:")
    print(f"  🤖 Transformer架构: 更强的序列建模能力")
    print(f"  🔧 神经架构搜索: 自动设计最优网络结构")
    print(f"  🎭 元学习: 快速适应新的符号回归任务")
    print(f"  🌐 图神经网络: 更好地表示表达式结构")
    print(f"  🔄 强化学习: 结合RL优化表达式搜索")
    
    # 性能评估
    if r2 > 0.8:
        print(f"\n🎉 深度学习成功学会了复杂的数学关系!")
        print(f"💡 网络找到了数据中的深层规律!")
    else:
        print(f"\n🔧 需要进一步优化:")
        print(f"  - 增加网络深度或宽度")
        print(f"  - 调整学习率和优化器")
        print(f"  - 增加训练数据")
        print(f"  - 使用预训练模型")

if __name__ == "__main__":
    main()
