# -*- coding: utf-8 -*-
"""
SISSO符号回归方法演示合集
========================

本文件夹包含了SISSO库中各种符号回归方法的详细演示代码。
每个演示都是独立的，可以单独运行。

演示文件列表：
============

🧬 **进化算法类**
├── genetic_programming_demo.py          # 遗传编程演示
├── ga_pso_hybrid_demo.py               # 遗传算法+PSO混合演示
└── island_gp_demo.py                   # 岛屿遗传编程演示

🎯 **稀疏建模类**  
├── sisso_basic_demo.py                 # SISSO基础演示
├── sisso_variants_demo.py              # SISSO变体演示
├── lasso_regression_demo.py            # LASSO稀疏回归演示
└── sindy_demo.py                       # SINDy动力学识别演示

🎲 **贝叶斯概率类**
├── bayesian_symbolic_regression_demo.py # 贝叶斯符号回归演示
└── probabilistic_program_induction_demo.py # 概率程序归纳演示

🤖 **强化学习类**
├── reinforcement_learning_sr_demo.py   # 强化学习符号回归演示
├── deep_symbolic_regression_demo.py    # 深度符号回归演示
└── neural_symbolic_hybrid_demo.py      # 神经符号混合演示

🔬 **混合新兴类**
├── evolutionary_gradient_demo.py       # 进化+梯度混合演示
├── physics_informed_sr_demo.py         # 物理约束符号回归演示
└── multi_objective_sr_demo.py          # 多目标符号回归演示

使用说明：
=========

1. **环境要求**：
   - Python 3.8+
   - sisso_py库
   - numpy, pandas, matplotlib, scikit-learn

2. **运行方式**：
   ```bash
   # 激活conda环境
   conda activate sisso-env
   
   # 运行单个演示
   python demo/genetic_programming_demo.py
   python demo/sisso_basic_demo.py
   python demo/physics_informed_sr_demo.py
   python demo/multi_objective_sr_demo.py
   python demo/island_gp_demo.py
   # ... 其他演示文件
   
   # 运行全部演示
   python run_all_demos.py
   ```

3. **演示特点**：
   - 每个演示都包含详细的原理解释
   - 提供可视化结果分析
   - 包含参数调优建议
   - 展示方法的优缺点和适用场景

4. **学习路径建议**：
   - 初学者：从SISSO基础演示开始
   - 进阶用户：尝试混合方法和新兴技术
   - 研究人员：关注贝叶斯和概率方法

演示数据说明：
=============

每个演示都使用精心设计的合成数据，模拟不同类型的数学关系：

- **线性关系**：基础的线性组合
- **多项式关系**：二次、三次等多项式
- **三角函数**：sin, cos等周期函数
- **指数对数**：exp, log等非线性函数
- **复合函数**：多种函数的组合
- **动力学系统**：微分方程和时间序列
- **噪声处理**：各种类型和强度的噪声

性能评估指标：
=============

所有演示都使用标准的回归评估指标：

- **MSE (均方误差)**：预测精度的基本指标
- **R² (决定系数)**：模型解释能力的度量
- **残差分析**：误差分布和模式检查
- **复杂度分析**：模型简洁性评估
- **收敛分析**：算法收敛速度和稳定性

可视化内容：
===========

每个演示都包含丰富的可视化分析：

1. **预测精度**：真实值vs预测值散点图
2. **残差分析**：残差分布和时间序列
3. **函数形状**：学到的函数与真实函数对比
4. **特征重要性**：各特征对目标的贡献
5. **学习曲线**：算法收敛过程可视化
6. **参数敏感性**：关键参数的影响分析

扩展建议：
=========

1. **自定义数据**：
   - 替换generate_demo_data()函数
   - 使用您自己的实际数据
   
2. **参数调优**：
   - 调整算法超参数
   - 对比不同配置的性能
   
3. **方法组合**：
   - 尝试多种方法的集成
   - 实现自定义的混合策略
   
4. **评估扩展**：
   - 添加领域特定的评估指标
   - 实现交叉验证和统计显著性检验

故障排除：
=========

常见问题及解决方案：

1. **导入错误**：确保sisso_py库正确安装
2. **内存问题**：减少数据量或调小参数
3. **收敛问题**：增加迭代次数或调整学习率
4. **精度问题**：检查数据质量和算法适用性

联系与反馈：
===========

如果您在使用过程中遇到问题或有改进建议，欢迎：
- 查看项目文档
- 提交GitHub issue
- 参与社区讨论

希望这些演示能帮助您更好地理解和应用符号回归技术！
"""

def main():
    """演示合集主函数"""
    print("🔬 SISSO符号回归方法演示合集")
    print("=" * 60)
    print("📁 本文件夹包含以下演示:")
    print()
    
    # 演示文件列表
    demos = {
        "🧬 进化算法类": [
            "genetic_programming_demo.py - 遗传编程演示",
            "ga_pso_hybrid_demo.py - 遗传算法+PSO混合演示",
            "island_gp_demo.py - 岛屿遗传编程演示"
        ],
        "🎯 稀疏建模类": [
            "sisso_basic_demo.py - SISSO基础演示", 
            "sisso_variants_demo.py - SISSO变体演示",
            "lasso_regression_demo.py - LASSO稀疏回归演示",
            "sindy_demo.py - SINDy动力学识别演示"
        ],
        "🎲 贝叶斯概率类": [
            "bayesian_symbolic_regression_demo.py - 贝叶斯符号回归演示",
            "probabilistic_program_induction_demo.py - 概率程序归纳演示"
        ],
        "🤖 强化学习类": [
            "reinforcement_learning_sr_demo.py - 强化学习符号回归演示",
            "deep_symbolic_regression_demo.py - 深度符号回归演示", 
            "neural_symbolic_hybrid_demo.py - 神经符号混合演示"
        ],
        "🔬 混合新兴类": [
            "evolutionary_gradient_demo.py - 进化+梯度混合演示",
            "physics_informed_sr_demo.py - 物理约束符号回归演示",
            "multi_objective_sr_demo.py - 多目标符号回归演示"
        ]
    }
    
    for category, demo_list in demos.items():
        print(f"{category}:")
        for demo in demo_list:
            print(f"  ├── {demo}")
        print()
    
    print("🚀 运行方式:")
    print("  python demo/<演示文件名>")
    print("  例如: python demo/sisso_basic_demo.py")
    print()
    print("📖 详细说明请查看各演示文件的文档字符串")
    print("💡 建议从SISSO基础演示开始学习")

if __name__ == "__main__":
    main()
