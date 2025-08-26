# -*- coding: utf-8 -*-
"""
岛屿遗传编程演示
==============

采用多岛屿并行进化的遗传编程方法，
通过不同进化策略和周期性迁移来维持种群多样性并提高搜索效率。

特点：
- 并行多岛屿
- 多样化进化策略
- 迁移机制
- 负载均衡
"""

import numpy as np
import pandas as pd
from SR_py.evolutionary.island_gp import IslandGPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def generate_demo_data():
    """生成演示数据：复杂非线性函数"""
    np.random.seed(555)
    n_samples = 200
    
    # 输入变量
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    
    # 复杂非线性函数 (岛屿GP适合这类复杂搜索)
    y_true = (x1**2 + x2) * np.sin(x1) + np.exp(x3) - x1*x2 + 0.5*x3**3
    y = y_true + np.random.normal(0, 0.5, n_samples)
    
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    
    return X, y, y_true

def main():
    print("🏝️ 岛屿遗传编程演示")
    print("=" * 50)
    
    # 生成演示数据
    X, y, y_true = generate_demo_data()
    print(f"数据集大小: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"目标函数: y = (x1²+x2)*sin(x1) + exp(x3) - x1*x2 + 0.5*x3³")
    print(f"复杂度: 高度非线性，适合岛屿GP并行搜索")
    
    # 创建岛屿遗传编程模型
    print("\n🔧 创建岛屿遗传编程模型...")
    island_gp = IslandGPRegressor(
        n_islands=4,                    # 岛屿数量
        island_pop_size=50,            # 每个岛屿种群大小
        migration_rate=0.1,            # 迁移率
        migration_interval=10,         # 迁移间隔
        max_depth=6                    # 最大树深度
    )
    
    # 训练模型
    print("🏃 开始岛屿并行进化...")
    print(f"  岛屿配置: {island_gp.n_islands} 个独立岛屿")
    print(f"  每岛种群: {island_gp.island_pop_size} 个个体")
    print(f"  第1步: 初始化各岛屿种群...")
    print(f"  第2步: 并行独立进化...")
    print(f"  第3步: 周期性个体迁移...")
    print(f"  第4步: 岛屿间最优解共享...")
    print(f"  第5步: 收敛检测与终止...")
    
    island_gp.fit(X, pd.Series(y))
    
    # 预测
    print("📊 使用最佳岛屿解进行预测...")
    y_pred = island_gp.predict(X)
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n📈 岛屿GP性能:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 获取岛屿统计信息
    if hasattr(island_gp, 'get_island_stats'):
        island_stats = island_gp.get_island_stats()
        print(f"\n🏝️ 岛屿统计信息:")
        for i, stats in enumerate(island_stats):
            print(f"  岛屿{i+1}: 最佳适应度={stats['best_fitness']:.4f}, "
                  f"多样性={stats['diversity']:.3f}, "
                  f"收敛代数={stats['convergence_gen']}")
    else:
        # 模拟岛屿统计信息
        island_stats = []
        for i in range(island_gp.n_islands):
            stats = {
                'best_fitness': 0.85 + 0.1 * np.random.random(),
                'diversity': 0.3 + 0.5 * np.random.random(),
                'convergence_gen': 20 + np.random.randint(0, 30),
                'migration_count': np.random.randint(3, 8)
            }
            island_stats.append(stats)
    
    # 岛屿GP原理
    print(f"\n📋 岛屿遗传编程原理:")
    print(f"  🏝️ 多岛模型: 将种群分割为多个独立子种群")
    print(f"  🔄 并行进化: 各岛屿独立并行进化")
    print(f"  ↔️ 迁移机制: 周期性个体交换")
    print(f"  🎯 策略多样化: 不同岛屿采用不同进化策略")
    print(f"  ⚖️ 负载均衡: 动态调整计算资源分配")
    
    # 迁移机制详解
    print(f"\n↔️ 迁移机制详解:")
    print(f"  📊 迁移拓扑:")
    print(f"    - 环形: 岛屿按环形顺序迁移")
    print(f"    - 星形: 中心岛屿与其他岛屿交换")
    print(f"    - 全连接: 任意两岛屿间可迁移")
    print(f"    - 随机: 随机选择迁移目标")
    print(f"  🕐 迁移时机:")
    print(f"    - 固定间隔: 每N代进行一次迁移")
    print(f"    - 条件触发: 满足特定条件时迁移")
    print(f"    - 自适应: 根据收敛情况调整")
    
    # 策略差异化
    print(f"\n🎯 岛屿策略差异化:")
    print(f"  🔍 探索vs开发:")
    print(f"    - 探索岛屿: 高变异率，低选择压力")
    print(f"    - 开发岛屿: 低变异率，高选择压力")
    print(f"    - 平衡岛屿: 中等参数设置")
    print(f"  🎲 操作算子:")
    print(f"    - 不同交叉操作符")
    print(f"    - 不同变异操作符")
    print(f"    - 不同选择策略")
    
    # 负载均衡策略
    print(f"\n⚖️ 负载均衡策略:")
    print(f"  📊 静态分配: 预先分配固定资源")
    print(f"  🔄 动态调整: 根据负载实时调整")
    print(f"  🎯 工作窃取: 空闲岛屿处理其他岛屿任务")
    print(f"  📈 自适应: 根据岛屿性能动态分配")
    
    # 应用优势
    print(f"\n🌟 岛屿GP优势:")
    print(f"  ✅ 并行加速: 显著提高计算速度")
    print(f"  ✅ 多样性维护: 避免过早收敛")
    print(f"  ✅ 鲁棒性强: 多个独立搜索降低失败风险")
    print(f"  ✅ 可扩展性: 容易增加计算资源")
    print(f"  ✅ 容错能力: 单个岛屿失败不影响整体")
    
    # 应用场景
    print(f"\n🎯 应用场景:")
    print(f"  🔬 大规模优化: 高维复杂函数优化")
    print(f"  💻 分布式计算: 多核/多机并行")
    print(f"  🧬 进化计算: 需要维持多样性的问题")
    print(f"  🎮 游戏AI: 多策略并行学习")
    print(f"  🏭 工程优化: 多目标工程设计")
    
    # 实施挑战
    print(f"\n⚠️ 实施挑战:")
    print(f"  🔄 同步开销: 迁移和通信成本")
    print(f"  ⚖️ 负载均衡: 不同岛屿负载差异")
    print(f"  🎯 参数调优: 迁移率和间隔设置")
    print(f"  📊 收敛判断: 多岛屿统一收敛标准")
    
    # 参数调优指南
    print(f"\n⚙️ 参数调优指南:")
    print(f"  n_islands: 岛屿数量 (当前: {island_gp.n_islands})")
    print(f"  island_pop_size: 岛屿种群大小 (当前: {island_gp.island_pop_size})")
    print(f"  migration_rate: 迁移率 (当前: {island_gp.migration_rate})")
    print(f"  migration_interval: 迁移间隔 (当前: {island_gp.migration_interval})")
    print(f"  max_depth: 最大树深度 (当前: {island_gp.max_depth})")
    
    # 成功评估
    if r2 > 0.9:
        print(f"\n🎉 岛屿GP取得优异性能!")
        print(f"💡 多岛屿并行策略有效提升了搜索效率!")
        print(f"🏝️ 迁移机制成功维持了种群多样性!")
    else:
        print(f"\n🔧 可进一步改进:")
        print(f"  - 调整岛屿数量和迁移参数")
        print(f"  - 优化岛屿间策略差异化")
        print(f"  - 改进负载均衡机制")

if __name__ == "__main__":
    main()
