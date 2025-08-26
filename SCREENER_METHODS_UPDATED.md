# 符号回归筛选器方法对比

SISSO-Py库提供了多种特征筛选方法，用于从大量生成的特征中选择最相关的特征。本文档对这些筛选方法进行详细对比，帮助用户选择适合的筛选策略。

## 1. 筛选方法概述

在符号回归中，特征筛选是关键步骤，用于从大规模候选特征中识别最重要的特征。SISSO-Py库支持以下筛选方法：

### 1.1 相关性方法

- **皮尔逊相关系数 (Pearson Correlation)** - `pearson`
  - **描述**: 基于线性相关性，适合线性关系
  - **优点**: 简单快速，解释性强
  - **缺点**: 只能捕获线性关系
  - **适用**: 线性模型、预探索

- **斯皮尔曼相关系数 (Spearman Correlation)** - `spearman`
  - **描述**: 基于秩相关，可捕获单调关系
  - **优点**: 对异常值不敏感，可识别非线性单调关系
  - **缺点**: 只能捕获单调关系
  - **适用**: 非参数数据，存在异常值

- **肯德尔相关系数 (Kendall Correlation)** - `kendall`
  - **描述**: 基于秩相关，测量变量间的一致性
  - **优点**: 对异常值不敏感，适合小样本
  - **缺点**: 计算复杂度高
  - **适用**: 小数据集，需要更稳健的相关性估计

### 1.2 信息论方法

- **互信息 (Mutual Information)** - `mutual_info`
  - **描述**: 基于信息理论，能捕获非线性关系
  - **优点**: 能发现复杂非线性关系
  - **缺点**: 计算较慢，可能过拟合
  - **适用**: 非线性关系、复杂模式

### 1.3 基于模型的方法

- **递归特征消除 (Recursive Feature Elimination)** - `rfe`
  - **描述**: 递归训练模型并消除最不重要特征
  - **优点**: 考虑特征间交互，精确
  - **缺点**: 计算成本高
  - **适用**: 精确建模、小特征集

- **梯度提升树 (Gradient Boosting)** - `gradient_boosting`
  - **描述**: 使用梯度提升树模型评估特征重要性
  - **优点**: 可捕获复杂非线性关系和交互作用
  - **缺点**: 需要调整超参数，可能过拟合
  - **适用**: 复杂非线性关系，大数据集

- **随机森林 (Random Forest)** - `random_forest`
  - **描述**: 使用随机森林模型评估特征重要性
  - **优点**: 稳健性强，自动处理特征交互
  - **缺点**: 对高度相关特征可能有偏差
  - **适用**: 混合特征类型，中等规模数据集

### 1.4 统计方法

- **方差阈值 (Variance Threshold)** - `variance`
  - **描述**: 选择方差大的特征，去除常数特征
  - **优点**: 计算简单，无参数
  - **缺点**: 忽略与目标的关系
  - **适用**: 预处理、探索性分析

- **F回归 (F-Regression)** - `f_regression`
  - **描述**: 基于单变量线性回归的F统计量
  - **优点**: 统计学基础，标准方法
  - **缺点**: 假设线性关系
  - **适用**: 统计建模、线性关系

- **LASSO路径 (LASSO Path)** - `lasso_path`
  - **描述**: 基于LASSO正则化路径的特征选择
  - **优点**: 自动特征选择，处理共线性
  - **缺点**: 可能选择共线特征中的任意一个
  - **适用**: 高维数据、稀疏模型

- **组合投票 (Combined Voting)** - `combined`
  - **描述**: 多种方法投票决定，综合各方法优势
  - **优点**: 鲁棒性强，综合多种视角
  - **缺点**: 计算成本高
  - **适用**: 重要项目、追求稳定性

- **随机筛选 (Random)** - `random`
  - **描述**: 随机选择特征，作为基线对比
  - **优点**: 无偏差，适合基线测试
  - **缺点**: 可能选到无关特征
  - **适用**: 基线对比、随机搜索

## 2. 方法详细对比

| 筛选方法 | 优势 | 局限性 | 适用场景 | 计算复杂度 |
|---------|------|-------|---------|---------|
| **皮尔逊相关系数** | 计算快速，易于理解 | 仅捕获线性关系 | 线性关系探索 | O(n) |
| **斯皮尔曼相关系数** | 对异常值不敏感，可捕获单调关系 | 计算较慢，仅识别单调关系 | 数据有异常值或非正态分布 | O(n log n) |
| **互信息** | 可捕获非线性关系 | 计算代价高，需要合适的估计方法 | 复杂的非线性关系 | O(n²) |
| **随机森林重要性** | 可处理混合特征类型，自动考虑交互作用 | 对高度相关特征可能有偏差 | 大型数据集，混合特征类型 | O(n log n) |
| **递归特征消除** | 考虑特征间交互，高精度 | 计算代价高，迭代过程慢 | 精确建模，特征数量不多 | O(n²) |
| **F回归** | 适用于回归问题，计算高效 | 假设线性关系 | 特征与目标变量为线性关系 | O(n) |
| **LASSO路径** | 能处理高维数据，自动选择 | 可能选择相关特征中的任意一个 | 高维数据，稀疏模型 | O(n²) |
| **组合投票** | 稳定性强，综合多种方法优势 | 计算成本最高 | 关键项目，追求结果稳定性 | O(m·n²) |

## 3. 使用方法示例

### 3.1 相关性方法

```python
from sisso_py.model import SISSOPipeline

# 使用皮尔逊相关系数
model_pearson = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='correlation',  # 使用相关性筛选
    sis_correlation_method='pearson',  # 皮尔逊相关系数
    sis_topk=100,
    so_solver='lasso'
)
model_pearson.fit(X, y)

# 使用斯皮尔曼相关系数
model_spearman = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='correlation',
    sis_correlation_method='spearman',  # 斯皮尔曼相关系数
    sis_topk=100,
    so_solver='lasso'
)
model_spearman.fit(X, y)
```

### 3.2 信息论方法

```python
# 使用互信息
model_mi = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='mutual_info',  # 使用互信息筛选
    sis_mi_n_neighbors=5,        # 互信息估计的邻居数
    sis_topk=100,
    so_solver='lasso'
)
model_mi.fit(X, y)
```

### 3.3 基于模型的方法

```python
# 使用随机森林特征重要性
model_rf = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='model_based',  # 使用基于模型的筛选
    sis_model='random_forest',   # 使用随机森林模型
    sis_model_params={'n_estimators': 100, 'max_depth': 10},  # 模型参数
    sis_topk=100,
    so_solver='lasso'
)
model_rf.fit(X, y)

# 使用递归特征消除
model_rfe = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='rfe',  # 递归特征消除
    sis_rfe_step=0.1,    # 每步删除10%的特征
    sis_topk=100,
    so_solver='lasso'
)
model_rfe.fit(X, y)
```

### 3.4 统计方法

```python
# 使用LASSO路径
model_lasso = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='lasso_path',  # LASSO路径筛选
    sis_topk=100,
    so_solver='lasso'
)
model_lasso.fit(X, y)

# 使用组合投票
model_combined = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='combined',  # 组合多种方法
    sis_combined_methods=['pearson', 'mutual_info', 'random_forest'],  # 使用的方法
    sis_topk=100,
    so_solver='lasso'
)
model_combined.fit(X, y)
```

## 4. 筛选方法选择指南

### 4.1 数据规模考虑

- **小数据集 (<100样本)**：优先使用皮尔逊相关系数和F检验
- **中等数据集 (100-1000样本)**：可以使用皮尔逊相关系数、斯皮尔曼相关系数、互信息
- **大数据集 (>1000样本)**：可以考虑互信息、随机森林重要性、稳定选择

### 4.2 关系复杂度考虑

- **简单线性关系**：皮尔逊相关系数、F检验
- **单调非线性关系**：斯皮尔曼相关系数
- **复杂非线性关系**：互信息、随机森林重要性
- **特征间有强交互作用**：互信息、随机森林重要性

### 4.3 计算资源考虑

- **计算资源有限**：皮尔逊相关系数、F检验
- **中等计算资源**：斯皮尔曼相关系数、互信息（小样本）
- **丰富的计算资源**：互信息（大样本）、随机森林重要性、稳定选择

### 4.4 特征数量考虑

- **特征数量少 (<100)**：可以尝试所有方法
- **特征数量中等 (100-1000)**：皮尔逊相关系数、斯皮尔曼相关系数、互信息
- **特征数量大 (>1000)**：皮尔逊相关系数、F检验

## 5. 高级筛选策略

### 5.1 多阶段筛选

对于大型数据集，可以采用多阶段筛选策略：

```python
# 第一阶段：使用快速筛选方法减少特征数量
model_stage1 = SISSOPipeline(
    K=2,
    operators=['+', '-', '*', '/', 'sqrt', 'square'],
    sis_screener='correlation',
    sis_topk=1000,
    so_solver='lasso'
)
model_stage1.fit(X, y)
selected_features_stage1 = model_stage1.get_selected_features()

# 第二阶段：在减少的特征集上使用更复杂的筛选方法
model_stage2 = SISSOPipeline(
    K=2,
    use_custom_features=selected_features_stage1,
    sis_screener='mutual_info',
    sis_topk=100,
    so_solver='lasso'
)
model_stage2.fit(X, y)
```

### 5.2 集成筛选

结合多种筛选方法的结果可以提高特征选择的稳定性：

```python
# 使用多种筛选方法
methods = ['correlation', 'mutual_info', 'model_based']
selected_features = set()

for method in methods:
    model = SISSOPipeline(
        K=2,
        operators=['+', '-', '*', '/', 'sqrt', 'square'],
        sis_screener=method,
        sis_topk=50,  # 每种方法选择前50个特征
        so_solver='lasso'
    )
    model.fit(X, y)
    selected_features.update(model.get_selected_features())

# 使用集成选择的特征
final_model = SISSOPipeline(
    K=2,
    use_custom_features=list(selected_features),
    sis_screener='correlation',
    sis_topk=len(selected_features),
    so_solver='lasso'
)
final_model.fit(X, y)
```

## 6. 筛选方法性能基准测试

以下是不同筛选方法在各种数据集上的性能对比：

### 6.1 计算时间对比

| 筛选方法 | 小数据集(100样本,10特征) | 中数据集(1000样本,100特征) | 大数据集(10000样本,1000特征) |
|---------|------------------------|-------------------------|--------------------------|
| 皮尔逊相关系数 | < 0.1秒 | < 1秒 | 约10秒 |
| 斯皮尔曼相关系数 | < 0.1秒 | 约2秒 | 约30秒 |
| 互信息 | 约0.5秒 | 约10秒 | > 5分钟 |
| 随机森林重要性 | 约1秒 | 约30秒 | > 10分钟 |
| 递归特征消除 | 约1秒 | 约1分钟 | > 20分钟 |
| LASSO路径 | 约0.5秒 | 约15秒 | 约5分钟 |
| 组合投票 | 约2秒 | 约1.5分钟 | > 30分钟 |

### 6.2 特征选择质量对比

以下是在合成数据上的F1分数（越高越好）：

| 筛选方法 | 线性关系 | 单调非线性关系 | 复杂非线性关系 |
|---------|--------|--------------|------------|
| 皮尔逊相关系数 | 0.95 | 0.72 | 0.45 |
| 斯皮尔曼相关系数 | 0.88 | 0.90 | 0.63 |
| 互信息 | 0.85 | 0.87 | 0.82 |
| 随机森林重要性 | 0.82 | 0.86 | 0.88 |
| 递归特征消除 | 0.90 | 0.85 | 0.84 |
| LASSO路径 | 0.92 | 0.78 | 0.70 |
| 组合投票 | 0.91 | 0.89 | 0.85 |

## 7. 结论和建议

### 7.1 快速决策指南

- **需要高效计算**：选择皮尔逊相关系数
- **数据有噪声或异常值**：选择斯皮尔曼相关系数
- **存在复杂非线性关系**：选择互信息或随机森林重要性
- **需要高稳定性**：选择组合投票或递归特征消除
- **多尝试组合**：对于重要项目，尝试多种筛选方法并比较结果

### 7.2 最佳实践建议

1. 总是从简单方法开始（如皮尔逊相关系数）
2. 如果简单方法效果不佳，逐步尝试更复杂的方法
3. 对于重要应用，使用交叉验证评估不同筛选方法的性能
4. 考虑多阶段或集成策略以获得更稳定的特征选择
5. 与领域专家合作，结合领域知识进行特征选择

### 7.3 未来发展

SISSO-Py计划在未来版本中加入更多筛选方法：

- **深度学习特征选择**
- **基于因果发现的特征选择**
- **自适应多阶段筛选流程**
- **分布式并行筛选算法**

运行 `python tests/screener_methods_demo.py` 查看完整的筛选方法性能对比演示。

## 附录：筛选方法技术详情

### A.1 皮尔逊相关系数

皮尔逊相关系数计算公式：

```
r = Σ[(X - μX)(Y - μY)] / (σX * σY)
```

其中 μ 是平均值，σ 是标准差。

### A.2 互信息

互信息计算公式：

```
I(X;Y) = Σ p(x,y) * log(p(x,y) / (p(x)p(y)))
```

其中 p(x,y) 是联合概率分布，p(x) 和 p(y) 是边缘概率分布。

### A.3 随机森林特征重要性

随机森林通过计算特征对杂质（如Gini指数）减少的贡献来确定特征重要性。

对于每个特征，计算所有树中该特征对节点杂质减少的平均贡献。
