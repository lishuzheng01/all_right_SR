# SISSO模型创建与训练指南

本指南基于SISSO-Py项目的实际代码示例，详细介绍如何创建和训练SISSO符号回归模型，重点关注参数选择策略和模型质量提升方法。

## 📋 目录

- [基础模型创建](#基础模型创建)
- [核心参数详解](#核心参数详解)
- [参数选择策略](#参数选择策略)
- [模型质量提升技巧](#模型质量提升技巧)
- [实际应用案例](#实际应用案例)
- [故障排除与优化](#故障排除与优化)
- [最佳实践总结](#最佳实践总结)

## 🚀 基础模型创建

### 最简单的模型创建

根据项目中的 `test01.py` 示例：

```python
import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

# 准备数据
X = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100)
})
y = 2 * X['x1']**2 + 3 * X['x2'] - X['x3'] + np.random.randn(100) * 0.1

# 创建和训练模型
model = SissoRegressor(
    K=2,  # 最大复杂度层数
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log'],
    sis_screener='random',  # 筛选方法
    sis_topk=1000,  # SIS 保留特征数
    so_solver='lasso',  # 稀疏求解器
    so_max_terms=2,  # 最终模型最大项数
    cv=5  # 交叉验证折数
)

# 拟合模型
model.fit(X, y)

# 预测和评估
y_pred = model.predict(X)
report = model.explain()
print(report)
```

### 改进版模型配置

基于 `test02_improved.py` 的优化配置：

```python
# 设置随机种子确保可重复性
np.random.seed(42)

# 准备更大的数据集
n_samples = 500  # 增加样本数量

# 简化目标函数，减少噪声
y = 2 * X['x1']**2 + 3 * X['x2'] - X['x3'] + np.random.randn(n_samples) * 0.05

# 创建更保守的模型配置
model = SissoRegressor(
    K=2,  # 降低复杂度
    operators=['+', '-', '*', 'safe_div', 'square'],  # 移除log和sqrt以避免数值问题
    sis_screener='pearson',  # 使用皮尔逊相关性筛选
    sis_topk=500,  # 减少特征数量
    so_solver='omp',  # 使用OMP求解器
    so_max_terms=3,  # 减少最终项数
    cv=5,
    random_state=42
)
```

## 🔧 核心参数详解

### K - 复杂度层数

**作用**：控制特征生成的深度，决定表达式的复杂度
**典型值**：1-5
**选择策略**：

```python
# 小数据集 (< 100 样本)
K = 2

# 中等数据集 (100-1000 样本)  
K = 3

# 大数据集 (> 1000 样本)
K = min(4, int(np.log2(n_features)) + 1)

# 示例：根据数据规模动态调整
n_samples, n_features = X.shape
if n_samples < 100:
    K = 2
elif n_samples < 1000:
    K = 3
else:
    K = min(4, int(np.log2(n_features)) + 1)
```

### operators - 操作符选择

**作用**：定义可用的数学操作符
**推荐配置**：

```python
# 基础配置 (稳定性优先)
basic_ops = ['+', '-', '*', 'safe_div', 'square']

# 标准配置 (平衡性能与复杂度)
standard_ops = ['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log']

# 扩展配置 (功能完整)
extended_ops = ['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log', 
                'exp', 'abs', 'sin', 'cos', 'reciprocal']

# 物理建模专用
physics_ops = ['+', '-', '*', 'safe_div', 'sqrt', 'square', 
               'reciprocal', 'poly2', 'poly3']

# 示例：根据问题类型选择
if problem_type == 'linear':
    operators = ['+', '-', '*', 'safe_div']
elif problem_type == 'polynomial':
    operators = ['+', '-', '*', 'safe_div', 'square', 'poly3']
elif problem_type == 'physics':
    operators = physics_ops
else:
    operators = standard_ops
```

### sis_screener - 特征筛选方法

根据 `SCREENER_METHODS.md`，有8种可选方法：

```python
# 推荐的筛选方法选择策略
screener_choice = {
    '探索阶段': 'pearson',           # 快速了解数据特征
    '线性关系': 'f_regression',       # 专门针对线性关系
    '非线性关系': 'mutual_info',      # 能捕获复杂模式  
    '高维数据': 'lasso_path',         # 处理特征维度灾难
    '稳健建模': 'combined',           # 多角度验证
    '精确建模': 'rfe',               # 递归特征消除
    '基线对比': 'random'             # 随机基线
}

# 材料科学示例 (来自 test_materials_bulk_modulus.py)
model = SissoRegressor(
    K=5,
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log', 'exp', 'abs', 'reciprocal'],
    sis_screener='mutual_info',  # 用于捕获复杂的非线性关系
    sis_topk=20,
    so_solver='lasso',
    so_max_terms=2,
    cv=5,
    random_state=42
)
```

### sis_topk - 保留特征数

**动态调整策略**：

```python
def calculate_optimal_topk(n_samples, n_features):
    """根据数据规模计算最优的特征保留数量"""
    if n_samples < 100:
        return min(500, n_samples * 10)
    elif n_samples < 1000:
        return min(2000, n_samples * 5)
    else:
        return min(5000, n_samples * 2)

# 使用示例
optimal_topk = calculate_optimal_topk(len(X), X.shape[1])
model = SissoRegressor(sis_topk=optimal_topk)
```

### so_solver - 稀疏求解器

**选择策略**：

```python
# OMP - 直接控制特征数量
model_omp = SissoRegressor(
    so_solver='omp',
    so_max_terms=3,  # 明确指定特征数
    cv=5
)

# Lasso - 自动特征选择
model_lasso = SissoRegressor(
    so_solver='lasso',
    cv=10  # 更多交叉验证折数
)

# ElasticNet - 平衡L1和L2正则化
model_en = SissoRegressor(
    so_solver='elasticnet',
    cv=5
)
```

## 📈 参数选择策略

### 基于数据特征的参数选择

```python
def adaptive_parameter_selection(X, y):
    """基于数据特征自适应选择参数"""
    n_samples, n_features = X.shape
    
    # 数据复杂度评估
    if n_features <= 5:
        complexity_level = 'simple'
    elif n_features <= 20:
        complexity_level = 'moderate'
    else:
        complexity_level = 'complex'
    
    # 样本规模评估
    if n_samples < 100:
        sample_size = 'small'
    elif n_samples < 1000:
        sample_size = 'medium'
    else:
        sample_size = 'large'
    
    # 参数配置矩阵
    config_matrix = {
        ('simple', 'small'): {
            'K': 2,
            'operators': ['+', '-', '*', 'safe_div', 'square'],
            'sis_screener': 'pearson',
            'sis_topk': 200,
            'so_solver': 'omp',
            'so_max_terms': 2
        },
        ('simple', 'medium'): {
            'K': 3,
            'operators': ['+', '-', '*', 'safe_div', 'sqrt', 'square'],
            'sis_screener': 'pearson',
            'sis_topk': 500,
            'so_solver': 'omp',
            'so_max_terms': 3
        },
        ('moderate', 'medium'): {
            'K': 3,
            'operators': ['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log'],
            'sis_screener': 'mutual_info',
            'sis_topk': 1000,
            'so_solver': 'lasso',
            'so_max_terms': 4
        },
        ('complex', 'large'): {
            'K': 4,
            'operators': ['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log', 'exp', 'abs'],
            'sis_screener': 'lasso_path',
            'sis_topk': 2000,
            'so_solver': 'elasticnet',
            'so_max_terms': 5
        }
    }
    
    key = (complexity_level, sample_size)
    if key in config_matrix:
        return config_matrix[key]
    else:
        # 默认配置
        return config_matrix[('moderate', 'medium')]

# 使用示例
config = adaptive_parameter_selection(X, y)
model = SissoRegressor(**config, cv=5, random_state=42)
```

### 渐进式参数调优

```python
def progressive_tuning(X, y, max_complexity=5):
    """渐进式参数调优"""
    best_model = None
    best_score = float('-inf')
    
    for K in range(2, max_complexity + 1):
        # 根据复杂度调整其他参数
        if K <= 2:
            sis_topk = min(500, len(X) * 3)
            so_max_terms = min(3, X.shape[1])
        elif K <= 3:
            sis_topk = min(1000, len(X) * 2)
            so_max_terms = min(5, X.shape[1])
        else:
            sis_topk = min(2000, len(X))
            so_max_terms = min(8, X.shape[1])
        
        model = SissoRegressor(
            K=K,
            sis_topk=sis_topk,
            so_max_terms=so_max_terms,
            cv=5,
            random_state=42
        )
        
        try:
            model.fit(X, y)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            avg_score = np.mean(scores)
            
            print(f"K={K}, TopK={sis_topk}, MaxTerms={so_max_terms}, R²={avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                
        except Exception as e:
            print(f"K={K} 训练失败: {e}")
            continue
    
    return best_model, best_score
```

## 💡 模型质量提升技巧

### 1. 数据预处理优化

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

def optimize_data_preprocessing(X, y):
    """优化数据预处理"""
    
    # 1. 处理缺失值
    X_clean = X.fillna(X.median())
    
    # 2. 移除常量列
    constant_cols = X_clean.columns[X_clean.nunique() <= 1]
    if len(constant_cols) > 0:
        print(f"移除常量列: {list(constant_cols)}")
        X_clean = X_clean.drop(constant_cols, axis=1)
    
    # 3. 移除高相关特征
    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        column for column in upper_tri.columns 
        if any(upper_tri[column] > 0.95)
    ]
    
    if high_corr_pairs:
        print(f"移除高相关特征: {high_corr_pairs}")
        X_clean = X_clean.drop(high_corr_pairs, axis=1)
    
    # 4. 鲁棒标准化
    scaler = RobustScaler()  # 对异常值更鲁棒
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )
    
    return X_scaled, y, scaler
```

### 2. 集成学习提升效果

```python
def ensemble_sisso_models(X, y):
    """使用集成学习提升SISSO模型效果"""
    from sklearn.ensemble import VotingRegressor
    
    # 创建多个不同配置的SISSO模型
    sisso1 = SissoRegressor(
        K=2, so_solver='omp', sis_screener='pearson', random_state=42
    )
    sisso2 = SissoRegressor(
        K=3, so_solver='lasso', sis_screener='mutual_info', random_state=43
    )
    sisso3 = SissoRegressor(
        K=2, so_solver='elasticnet', sis_screener='lasso_path', random_state=44
    )
    
    # 创建投票回归器
    ensemble = VotingRegressor([
        ('sisso_omp', sisso1),
        ('sisso_lasso', sisso2),
        ('sisso_en', sisso3)
    ])
    
    # 训练集成模型
    ensemble.fit(X, y)
    return ensemble
```

### 3. 交叉验证策略优化

```python
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold

def optimize_cross_validation(X, y, data_type='regression'):
    """优化交叉验证策略"""
    
    if data_type == 'time_series':
        # 时间序列数据使用TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
    elif len(y) < 100:
        # 小数据集使用Leave-One-Out或较少折数
        cv = min(5, len(y) // 10)
    else:
        # 标准回归使用KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    model = SissoRegressor(cv=cv)
    return model
```

### 4. 模型验证与鲁棒性检查

```python
def validate_model_robustness(model, X, y, n_bootstrap=50):
    """验证模型鲁棒性"""
    from sklearn.utils import resample
    from sklearn.metrics import r2_score
    from collections import Counter
    
    results = {
        'formulas': [],
        'r2_scores': [],
        'coefficients': []
    }
    
    print("进行Bootstrap验证...")
    
    for i in range(n_bootstrap):
        # 重采样
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # 使用相同配置训练新模型
        boot_model = SissoRegressor(
            K=model.K,
            operators=model.operators,
            sis_screener=model.sis_screener,
            sis_topk=model.sis_topk,
            so_solver=model.so_solver,
            so_max_terms=model.so_max_terms,
            cv=model.cv,
            random_state=i
        )
        
        try:
            boot_model.fit(X_boot, y_boot)
            y_pred = boot_model.predict(X_boot)
            r2 = r2_score(y_boot, y_pred)
            
            report = boot_model.explain()
            formula = report['results']['final_model']['formula_latex']
            
            results['formulas'].append(formula)
            results['r2_scores'].append(r2)
            
        except Exception:
            continue
    
    # 分析结果
    print(f"\n=== Bootstrap验证结果 (n={len(results['r2_scores'])}) ===")
    print(f"R² 平均值: {np.mean(results['r2_scores']):.4f} ± {np.std(results['r2_scores']):.4f}")
    
    # 统计公式出现频率
    formula_counts = Counter(results['formulas'])
    print(f"\n最常见的公式:")
    for formula, count in formula_counts.most_common(3):
        print(f"  {count:3d}x: {formula}")
    
    return results
```

## 🎯 实际应用案例

### 案例1：材料科学 - 体积模量预测

基于 `test_materials_bulk_modulus.py`：

```python
def materials_science_config():
    """材料科学专用配置"""
    return SissoRegressor(
        K=5,  # 较高复杂度捕获物理关系
        operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log', 'exp', 'abs', 'reciprocal'],
        sis_screener='mutual_info',  # 捕获非线性物理关系
        sis_topk=20,  # 材料数据通常特征数较少
        so_solver='lasso',  # 自动特征选择
        so_max_terms=2,  # 保持公式简洁
        cv=5,
        random_state=42
    )
```

### 案例2：简单函数拟合

基于 `test03.py`：

```python
def simple_function_fitting():
    """简单函数拟合配置"""
    return SissoRegressor(
        K=1,  # 低复杂度
        operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log', 'exp', 'abs', 
                  'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'reciprocal'],
        sis_screener='random',
        sis_topk=1000,
        so_solver='omp',  # 直接控制项数
        so_max_terms=3,
        cv=5
    )
```

## 🛠️ 故障排除与优化

### 常见问题及解决方案

#### 1. 内存不足

```python
# 解决方案：减少特征生成数量
model = SissoRegressor(
    K=2,                # 降低层数
    sis_topk=500,       # 减少筛选特征数
    so_max_terms=3      # 限制最终模型复杂度
)
```

#### 2. 收敛问题

```python
# 解决方案：尝试不同求解器
model = SissoRegressor(
    so_solver='lasso',     # 或 'elasticnet'
    cv=10,                 # 增加交叉验证折数
    random_state=42        # 固定随机种子
)
```

#### 3. 过拟合问题

```python
# 解决方案：增加正则化
model = SissoRegressor(
    so_max_terms=2,        # 减少最终项数
    cv=10,                 # 更严格的交叉验证
    sis_topk=500          # 减少候选特征数
)
```

### 性能优化

#### 并行计算

```python
model = SissoRegressor(
    n_jobs=-1,  # 使用所有CPU核心
    # n_jobs=4   # 或指定核心数
)
```

#### 内存优化

```python
def memory_efficient_training(X, y):
    """内存优化训练"""
    n_samples = len(X)
    
    if n_samples > 5000:
        # 大数据集：先用小样本探索
        sample_idx = np.random.choice(n_samples, 2000, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        # 小模型探索
        small_model = SissoRegressor(K=2, sis_topk=500, so_max_terms=3)
        small_model.fit(X_sample, y_sample)
        
        # 在全数据集上使用保守参数
        final_model = SissoRegressor(K=3, sis_topk=1000, so_max_terms=5)
        final_model.fit(X, y)
        
        return final_model
    else:
        # 小数据集：正常训练
        model = SissoRegressor()
        model.fit(X, y)
        return model
```

## 📊 最佳实践总结

### 参数选择决策树

```
数据规模
├── 小数据集 (< 100 样本)
│   ├── K = 2
│   ├── sis_topk = 200-500
│   ├── so_max_terms = 2-3
│   └── sis_screener = 'pearson'
│
├── 中等数据集 (100-1000 样本)
│   ├── K = 3
│   ├── sis_topk = 500-1000
│   ├── so_max_terms = 3-5
│   └── sis_screener = 'mutual_info'
│
└── 大数据集 (> 1000 样本)
    ├── K = 3-4
    ├── sis_topk = 1000-2000
    ├── so_max_terms = 5-8
    └── sis_screener = 'lasso_path'
```

### 质量提升检查清单

- [ ] **数据预处理**
  - [ ] 处理缺失值
  - [ ] 移除常量特征
  - [ ] 处理高相关特征
  - [ ] 适当的数据标准化

- [ ] **参数优化**
  - [ ] 根据数据规模选择K值
  - [ ] 合理选择操作符集合
  - [ ] 适配筛选方法
  - [ ] 动态调整sis_topk

- [ ] **模型验证**
  - [ ] 交叉验证配置
  - [ ] Bootstrap鲁棒性检查
  - [ ] 多配置对比
  - [ ] 集成学习考虑

- [ ] **结果分析**
  - [ ] 公式合理性检查
  - [ ] 物理意义验证（如适用）
  - [ ] 泛化能力评估
  - [ ] 复杂度vs性能权衡

### 推荐的工作流程

1. **数据探索阶段**

   ```python
   # 快速探索
   model_explore = SissoRegressor(
       K=2, sis_screener='pearson', sis_topk=500, so_max_terms=3
   )
   ```

2. **深入建模阶段**

   ```python
   # 精确建模
   model_precise = SissoRegressor(
       K=3, sis_screener='mutual_info', sis_topk=1000, so_max_terms=5
   )
   ```

3. **最终验证阶段**

   ```python
   # 鲁棒验证
   model_robust = SissoRegressor(
       K=3, sis_screener='combined', cv=10, so_solver='elasticnet'
   )
   ```

通过遵循这些指导原则和最佳实践，您可以显著提升SISSO模型的质量和可靠性，发现更准确、更有物理意义的符号公式。

---

## 📝 参考文件

- `test01.py` - 基础模型示例
- `test02_improved.py` - 改进配置示例  
- `test_materials_bulk_modulus.py` - 材料科学应用案例
- `SCREENER_METHODS.md` - 特征筛选方法指南
- `README.md` - 详细API文档
