# SISSO-Py 特征筛选方法指南

SISSO-Py 现在支持 **8种不同的特征筛选方法**，您可以根据数据特点和需求选择最适合的方法。

## 🔍 可用的筛选方法

### 1. `pearson` - Pearson相关系数

- **描述**: 基于线性相关性，适合线性关系
- **优点**: 简单快速，解释性强
- **缺点**: 只能捕获线性关系
- **适用**: 线性模型、预探索

### 2. `mutual_info` - 互信息

- **描述**: 基于信息理论，能捕获非线性关系
- **优点**: 能发现复杂非线性关系
- **缺点**: 计算较慢，可能过拟合
- **适用**: 非线性关系、复杂模式

### 3. `random` - 随机筛选

- **描述**: 随机选择特征，作为基线对比
- **优点**: 无偏差，适合基线测试
- **缺点**: 可能选到无关特征
- **适用**: 基线对比、随机搜索

### 4. `variance` - 方差筛选

- **描述**: 选择方差大的特征，去除常数特征
- **优点**: 快速去除无变化特征
- **缺点**: 忽略与目标的关系
- **适用**: 预处理、去除常数特征

### 5. `f_regression` - F统计量

- **描述**: 基于单变量线性回归的F统计量
- **优点**: 统计学基础，标准方法
- **缺点**: 假设线性关系
- **适用**: 统计建模、线性关系

### 6. `rfe` - 递归特征消除

- **描述**: 递归训练模型并消除最不重要特征
- **优点**: 考虑特征间交互，精确
- **缺点**: 计算成本高
- **适用**: 精确建模、小特征集

### 7. `lasso_path` - LASSO路径

- **描述**: 基于LASSO正则化路径的特征选择
- **优点**: 自动特征选择，处理共线性
- **缺点**: 可能选择共线特征中的任意一个
- **适用**: 高维数据、稀疏模型

### 8. `combined` - 组合投票

- **描述**: 多种方法投票决定，综合各方法优势
- **优点**: 鲁棒性强，综合多种视角
- **缺点**: 计算成本高
- **适用**: 重要项目、追求稳定性

## 🚀 使用方法

```python
from sisso_py import SissoRegressor

# 选择筛选方法
model = SissoRegressor(
    K=2,
    operators=['+', '-', '*', 'square'],
    sis_screener='lasso_path',  # 选择筛选方法
    sis_topk=100,
    so_solver='omp',
    so_max_terms=3
)

model.fit(X, y)
report = model.explain()
print(report)
```

## 💡 选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **探索阶段** | `pearson`, `mutual_info` | 快速了解数据特征 |
| **线性关系** | `pearson`, `f_regression` | 专门针对线性关系优化 |
| **非线性关系** | `mutual_info`, `lasso_path` | 能捕获复杂模式 |
| **高维数据** | `lasso_path`, `variance` | 处理特征维度灾难 |
| **稳健建模** | `combined`, `rfe` | 多角度验证，结果更可靠 |
| **基线对比** | `random` | 提供随机基线参考 |

## 🧪 性能对比

运行 `python screener_methods_demo.py` 查看完整的性能对比演示。

## 🔧 高级用法

```python
# 组合不同筛选方法进行对比
methods = ['pearson', 'mutual_info', 'lasso_path', 'combined']

for method in methods:
    model = SissoRegressor(sis_screener=method, ...)
    model.fit(X, y)
    report = model.explain()
    print(f"{method}: R² = {report['results']['metrics']['train_r2']:.4f}")
```

## 📊 实验结果

根据我们的测试，不同方法在不同场景下的表现：

- **`rfe`** 和 **`lasso_path`** 在复杂非线性关系中表现最佳
- **`pearson`** 在简单线性关系中快速有效  
- **`combined`** 提供最稳定的结果
- **`random`** 作为基线，帮助评估其他方法的有效性

## 🎯 最佳实践

1. **开始探索**: 用 `pearson` 快速了解数据
2. **深入分析**: 用 `mutual_info` 或 `lasso_path` 找复杂关系
3. **精确建模**: 用 `rfe` 或 `combined` 获得最佳结果
4. **验证有效性**: 用 `random` 作为基线对比

现在SISSO-Py为您提供了强大而灵活的特征筛选工具箱！🎉
