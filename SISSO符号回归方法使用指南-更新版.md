# SISSO-Py 符号回归方法使用指南

本指南提供了SISSO-Py库中实现的各类符号回归方法的使用教程，帮助用户根据具体场景选择和使用合适的符号回归技术。

## 1. 符号回归概述

符号回归是一种机器学习方法，旨在从数据中发现最能描述因变量与自变量关系的数学表达式。与传统回归方法不同，符号回归不仅能找到变量之间的关系，还能以解析表达式的形式呈现，提供更好的可解释性。

SISSO-Py库提供了三大类符号回归方法：

1. **基于进化搜索的方法**：通过模拟自然进化过程，在表达式空间中搜索最优解
2. **基于稀疏回归的方法**：生成大量候选特征，然后使用稀疏回归选择重要特征
3. **基于概率建模的方法**：使用贝叶斯方法为表达式分配概率，提供不确定性估计

## 2. 数据准备

所有方法都需要准备好的输入数据，通常是pandas DataFrame格式：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 准备数据
X = pd.DataFrame({
    'x1': np.random.random(100),
    'x2': np.random.random(100),
    'x3': np.random.random(100)
})

# 假设真实函数为 y = x1^2 + sin(x2) - x3
y = X['x1']**2 + np.sin(X['x2']) - X['x3']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 3. 使用进化搜索方法

### 3.1 遗传编程(GP)

遗传编程是符号回归中最常用的方法之一，通过模拟自然进化过程来发现最优表达式。

```python
from sisso_py.evolutionary import GeneticProgrammingRegressor

# 创建模型
gp = GeneticProgrammingRegressor(
    population_size=100,       # 种群大小
    generations=50,            # 进化代数
    operators=['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt'], # 操作符集
    parsimony_coefficient=0.1, # 复杂度惩罚系数
    tournament_size=7,         # 锦标赛选择大小
    random_state=42            # 随机种子
)

# 训练模型
gp.fit(X_train, y_train)

# 预测
y_pred = gp.predict(X_test)

# 评估模型
r2 = gp.score(X_test, y_test)
print(f"测试集R²: {r2}")

# 获取表达式
try:
    # 尝试不同可能的属性
    if hasattr(gp, 'best_program_'):
        formula = str(gp.best_program_)
    elif hasattr(gp, '_program'):
        formula = str(gp._program)
    else:
        formula = "无法获取表达式"
    print(f"最佳表达式: {formula}")
except:
    print("无法获取表达式")
```

### 3.2 岛模型遗传编程

岛模型通过在多个相对隔离的种群中并行进化来保持多样性，有助于防止过早收敛到局部最优解。

```python
from sisso_py.evolutionary import IslandGPRegressor

# 创建模型
island_gp = IslandGPRegressor(
    n_islands=5,               # 岛屿数量
    island_size=40,            # 每个岛的种群大小
    migration_freq=5,          # 迁移频率(多少代进行一次)
    migration_size=3,          # 每次迁移的个体数
    operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'],
    random_state=42
)

# 训练模型
island_gp.fit(X_train, y_train)

# 预测和评估
y_pred = island_gp.predict(X_test)
r2 = island_gp.score(X_test, y_test)
print(f"测试集R²: {r2}")
```

## 4. 使用稀疏回归方法

### 4.1 SISSO

SISSO (确定性筛选与稀疏回归)是为处理大规模特征空间而设计的方法，特别适用于材料科学等领域。

```python
from sisso_py.sparse_regression import SISSORegressor

# 创建模型
sisso = SISSORegressor(
    K=3,                        # 特征复杂度层级
    operators=['+', '-', '*', '/', 'sqrt', 'exp', 'log'],
    sis_screener='pearson',     # 筛选方法: 'pearson', 'spearman', 'mutual_info'
    sis_topk=500,               # 筛选阶段保留的特征数
    so_solver='lasso',          # 稀疏算子: 'lasso', 'ridge', 'omp'
    so_max_terms=3,             # 最终模型最大特征数
    dimensional_check=False,    # 是否检查物理量纲一致性
    random_state=42
)

# 训练模型
sisso.fit(X_train, y_train)

# 预测和评估
y_pred = sisso.predict(X_test)
r2 = sisso.score(X_test, y_test)

# 获取模型信息
model_info = sisso.get_model_info()
print(f"模型公式: {model_info.get('formula')}")
print(f"测试集R²: {r2}")
```

### 4.2 SINDy

SINDy(稀疏识别非线性动力学)适用于动力系统建模，能有效发现物理规律。

```python
from sisso_py.sparse_regression import SINDyRegressor

# 创建模型
sindy = SINDyRegressor(
    poly_degree=3,              # 多项式最高次数
    threshold=0.05,             # 稀疏阈值
    alpha=0.1,                  # 正则化参数
    random_state=42
)

# 训练模型
sindy.fit(X_train, y_train)

# 预测和评估
y_pred = sindy.predict(X_test)
r2 = sindy.score(X_test, y_test)

# 获取模型信息
try:
    model_info = sindy.get_model_info()
    print(f"模型公式: {model_info.get('formula')}")
except:
    print("无法获取模型信息")

print(f"测试集R²: {r2}")
```

### 4.3 Lasso特征回归

使用Lasso正则化从预先生成的特征中选择重要特征，适用于需要稀疏解的场景。

```python
from sisso_py.sparse_regression import LassoRegressor

# 创建模型
lasso = LassoRegressor(
    alpha=0.01,                  # 正则化参数
    poly_degree=2,               # 多项式特征最高次数
    random_state=42
)

# 训练模型
lasso.fit(X_train, y_train)

# 预测和评估
y_pred = lasso.predict(X_test)
r2 = lasso.score(X_test, y_test)

# 获取模型信息
try:
    model_info = lasso.get_model_info()
    print(f"模型公式: {model_info.get('formula')}")
except:
    print("无法获取模型信息")

print(f"测试集R²: {r2}")
```

## 5. 使用概率建模方法

### 5.1 贝叶斯符号回归(BSR)

BSR使用马尔可夫链蒙特卡洛采样对表达式空间进行探索，并提供不确定性估计。

```python
from sisso_py.probabilistic import BayesianSymbolicRegressor

# 创建模型
bsr = BayesianSymbolicRegressor(
    n_iter=1000,                # MCMC迭代次数
    burn_in=200,                # burn-in期
    operators=['+', '-', '*', '/', 'sin', 'exp', 'log'],
    random_state=42
)

# 训练模型
bsr.fit(X_train, y_train)

# 预测和评估
y_pred = bsr.predict(X_test)
r2 = bsr.score(X_test, y_test)

# 获取模型信息
try:
    model_info = bsr.get_model_info()
    print(f"最佳表达式: {model_info.get('formula')}")
    
    # 获取多个表达式及其后验概率
    if hasattr(bsr, 'get_top_expressions'):
        print("表达式后验概率:")
        for expr, prob in bsr.get_top_expressions(3):
            print(f" - {expr}: {prob:.4f}")
except:
    print("无法获取模型信息")

print(f"测试集R²: {r2}")
```

### 5.2 概率程序归纳(PPI)

PPI通过学习概率上下文无关语法来生成表达式，特别适合融入领域知识。

```python
from sisso_py.probabilistic import ProbabilisticProgramInduction

# 创建模型
ppi = ProbabilisticProgramInduction(
    n_iterations=100,           # 迭代次数
    population_size=50,         # 种群大小
    operators=['+', '-', '*', '/', 'sin', 'cos', 'exp'],
    random_state=42
)

# 训练模型
ppi.fit(X_train, y_train)

# 预测和评估
y_pred = ppi.predict(X_test)
r2 = ppi.score(X_test, y_test)

# 获取模型信息
try:
    model_info = ppi.get_model_info()
    print(f"最佳表达式: {model_info.get('formula')}")
except:
    print("无法获取模型信息")

print(f"测试集R²: {r2}")
```

## 6. 交叉验证与超参数调优

为获得更稳健的模型，可使用交叉验证和超参数调优：

```python
from sklearn.model_selection import GridSearchCV
from sisso_py.sparse_regression import SISSORegressor

# 定义参数网格
param_grid = {
    'K': [2, 3],
    'sis_topk': [100, 500],
    'so_max_terms': [2, 3, 4]
}

# 创建基础模型
base_model = SISSORegressor(
    operators=['+', '-', '*', '/'],
    sis_screener='pearson',
    so_solver='lasso',
    random_state=42
)

# 创建网格搜索
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

# 训练
grid_search.fit(X_train, y_train)

# 最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_}")

# 使用最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"测试集R²: {best_model.score(X_test, y_test)}")
```

## 7. 模型集成

不同符号回归方法可以通过集成进一步提高性能：

```python
import numpy as np
from sklearn.metrics import r2_score

# 训练多个模型
models = [
    ("GP", GeneticProgrammingRegressor(random_state=42)),
    ("SISSO", SISSORegressor(random_state=42)),
    ("BSR", BayesianSymbolicRegressor(random_state=42))
]

# 训练模型并获取预测
predictions = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    print(f"{name} 测试集R²: {r2_score(y_test, y_pred)}")

# 简单平均集成
ensemble_pred = np.mean(predictions, axis=0)
ensemble_r2 = r2_score(y_test, ensemble_pred)
print(f"集成模型测试集R²: {ensemble_r2}")
```

## 8. 物理量纲与约束

在物理和工程问题中，可以考虑量纲一致性约束：

```python
from sisso_py.dsl.dimension import Dimension
import pandas as pd

# 定义特征量纲
dimensions = {
    'mass': Dimension(M=1),
    'length': Dimension(L=1),
    'time': Dimension(T=1)
}

# 定义目标量纲(例如，能量的量纲: M*L²/T²)
target_dim = Dimension(M=1, L=2, T=-2)

# 创建数据
data = pd.DataFrame({
    'mass': [1, 2, 3, 4, 5],
    'length': [10, 20, 30, 40, 50],
    'time': [1, 2, 1, 3, 2]
})
y = 0.5 * data['mass'] * (data['length']/data['time'])**2  # 动能公式: 1/2 * m * v²

# 创建并训练模型
sisso = SISSORegressor(
    K=2,
    operators=['+', '-', '*', '/'],
    sis_topk=50,
    dimensional_check=True,  # 启用量纲检查
    random_state=42
)

# 训练模型时提供量纲信息
sisso.fit(
    data, y,
    feature_dimensions=dimensions,
    target_dimension=target_dim
)

# 获取符合物理量纲的表达式
model_info = sisso.get_model_info()
print(f"符合物理量纲的公式: {model_info.get('formula')}")
```

## 9. 大规模数据处理

对于大型数据集，可以使用以下策略：

```python
from sisso_py.sparse_regression import SISSORegressor

# 启用并行计算
sisso = SISSORegressor(
    K=2,
    sis_topk=1000,
    n_jobs=-1,  # 使用所有可用CPU
    random_state=42
)

# 对于非常大的数据集，可以先用子样本训练初步模型
from sklearn.model_selection import train_test_split

# 采样子集进行初步特征筛选
X_sub, _, y_sub, _ = train_test_split(
    X_large, y_large, 
    train_size=10000,  # 使用有限样本
    random_state=42
)

# 先在子集上训练
sisso.fit(X_sub, y_sub)

# 获取重要特征
model_info = sisso.get_model_info()
features = model_info.get('selected_features', [])

print(f"发现的重要特征: {features}")
```

## 10. 结果可视化

可视化模型结果有助于更好地理解和解释：

```python
import matplotlib.pyplot as plt
import numpy as np

# 假设是单变量模型，创建预测网格
x_grid = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
x_grid_df = pd.DataFrame(x_grid, columns=['x'])
y_grid = model.predict(x_grid_df)

# 绘制真实值vs预测值
plt.figure(figsize=(12, 5))

# 散点图
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, c='blue', label='真实值')
plt.scatter(X_test, y_pred, c='red', alpha=0.5, label='预测值')
plt.legend()
plt.title('真实值 vs. 预测值')

# 拟合曲线
plt.subplot(1, 2, 2)
plt.plot(x_grid, y_grid, 'g-', linewidth=2, label='模型')
plt.scatter(X_test, y_test, c='blue', alpha=0.5, label='测试集')
plt.legend()
plt.title(f'模型公式: {model_info.get("formula")}')

plt.tight_layout()
plt.show()

# 残差分析
plt.figure(figsize=(8, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
plt.show()
```

## 11. 最佳实践建议

1. **方法选择**:
   - 对于小型数据集和复杂关系，尝试进化方法(GP/岛模型GP)
   - 对于中大型数据集和较简单关系，尝试稀疏回归方法(SISSO/SINDy)
   - 需要不确定性估计时，使用贝叶斯方法(BSR)

2. **操作符选择**:
   - 根据领域知识选择合适的操作符集
   - 开始时可以使用较小的操作符集，然后逐渐扩展
   - 复杂问题可能需要更丰富的操作符，但要注意过拟合风险

3. **复杂度控制**:
   - 进化方法: 调整parsimony_coefficient参数
   - SISSO: 调整K(特征复杂度)和so_max_terms(最终模型项数)
   - 贝叶斯方法: 设置适当的先验分布

4. **模型评估**:
   - 使用交叉验证评估模型稳定性
   - 检查模型在不同测试集上的性能
   - 考虑模型复杂度与性能的权衡

5. **结果验证**:
   - 与理论模型或已知规律比较
   - 检查边界条件和极限行为
   - 使用领域知识验证表达式合理性

## 12. 故障排除

1. **内存错误**:
   - 降低特征复杂度(K)
   - 减小sis_topk参数
   - 使用更有针对性的操作符集

2. **过拟合**:
   - 增加正则化强度
   - 限制表达式复杂度
   - 使用更多训练数据

3. **表达式过于复杂**:
   - 增加parsimony_coefficient
   - 减少so_max_terms
   - 尝试更简单的操作符集

4. **计算时间过长**:
   - 减少迭代次数/世代数
   - 启用并行计算(n_jobs=-1)
   - 使用较小的初始特征集

## 13. 总结

SISSO-Py库提供了多种符号回归方法，每种方法都有其特定的优势和适用场景。在实际应用中，建议尝试多种方法并比较结果，根据具体需求选择最适合的方法。

符号回归不仅能提供准确的预测，还能发现可解释的数学关系，使其在科学发现、工程建模和数据分析等领域具有广泛应用价值。通过本指南介绍的技术，用户可以充分利用SISSO-Py库的功能，从数据中挖掘有意义的数学关系。
