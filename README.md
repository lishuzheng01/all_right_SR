# AllRight-SR: 多策略符号回归工具箱

## 项目说明
AllRight-SR 集成了多种符号回归算法，包括进化算法、稀疏建模、贝叶斯推断、强化学习以及多种混合优化方法，帮助用户从数据中自动发现符合物理意义的解析公式。

## 使用方法
以下示例默认已准备好训练数据 `X` (pandas.DataFrame) 和目标 `y` (pandas.Series)。

### 进化算法类
#### 遗传编程 (GP)
```python
from sisso_py.evolutionary.gp import GeneticProgramming
model = GeneticProgramming(population_size=50, n_generations=10)
model.fit(X, y, feature_names=X.columns)
pred = model.predict(X)
```

#### 遗传算法+PSO混合
```python
from sisso_py.evolutionary.ga_pso import GAPSORegressor
model = GAPSORegressor(generations=30)
model.fit(X, y)
pred = model.predict(X)
```

### 稀疏建模类
#### SISSO 基础
```python
from sisso_py.sparse_regression.sisso import SISSORegressor
model = SISSORegressor(K=3, sis_screener='pearson', so_solver='omp')
model.fit(X, y)
```

#### SISSO 筛选器: pearson
```python
SISSORegressor(sis_screener='pearson').fit(X, y)
```

#### SISSO 筛选器: f_regression
```python
SISSORegressor(sis_screener='f_regression').fit(X, y)
```

#### SISSO 筛选器: mutual_info
```python
SISSORegressor(sis_screener='mutual_info').fit(X, y)
```

#### SISSO 求解器: omp
```python
SISSORegressor(so_solver='omp').fit(X, y)
```

#### SISSO 求解器: lasso
```python
SISSORegressor(so_solver='lasso').fit(X, y)
```

#### SISSO 求解器: elasticnet
```python
SISSORegressor(so_solver='elasticnet').fit(X, y)
```

#### SISSO 维度检查
```python
from sisso_py.dsl.dimension import Dimension
dims = {'x1': Dimension([1,0,0,0,0,0,0])}
target_dim = Dimension([1,0,0,0,0,0,0])
SISSORegressor(dimensional_check=True).fit(X, y, feature_dimensions=dims, target_dimension=target_dim)
```

#### LASSO稀疏回归
```python
from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor
model = LassoRegressor(alpha=0.01)
model.fit(X, y)
```

#### SINDy
```python
from sisso_py.sparse_regression.sindy import SINDyRegressor
model = SINDyRegressor(poly_degree=3)
model.fit(X, y)
equation = model.get_equation()
```

### 贝叶斯概率类
#### 贝叶斯符号回归 (MCMC)
```python
from sisso_py.probabilistic.bsr import BayesianSymbolicRegressor
model = BayesianSymbolicRegressor(n_iter=2000)
model.fit(X, y)
info = model.get_model_info()
```

#### 概率程序归纳 (PCFG)
```python
from sisso_py.probabilistic.ppi import ProbabilisticProgramInduction
model = ProbabilisticProgramInduction(n_iterations=500)
model.fit(X, y)
info = model.get_model_info()
```

### 强化学习类
#### 强化学习符号回归
```python
from sisso_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
model = ReinforcementSymbolicRegression(max_episodes=50)
model.fit(X.values, y.values, feature_names=X.columns)
```

#### 深度符号回归
```python
from sisso_py.neural_symbolic.deep_sr import DeepSymbolicRegression
model = DeepSymbolicRegression(epochs=20)
model.fit(X.values, y.values, feature_names=X.columns)
```

#### 神经符号混合
```python
from sisso_py.neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
model = NeuralSymbolicHybrid(symbolic_component='gp')
model.fit(X.values, y.values, feature_names=X.columns)
```

### 混合新兴类
#### 进化+梯度混合
```python
from sisso_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
model = EvolutionaryGradientHybrid(evolution_phase_generations=10)
model.fit(X.values, y.values, feature_names=X.columns)
```

#### 物理约束符号回归
```python
from sisso_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
model = PhysicsInformedSymbolicRegression(dimensional_analysis=False)
model.fit(X.values, y.values, feature_names=X.columns)
```

#### 多目标符号回归
```python
from sisso_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
model = MultiObjectiveSymbolicRegression(n_generations=10)
model.fit(X.values, y.values, feature_names=X.columns)
```

---

# SISSO-Py: Python Implementation of Sure Independence Screening and Sparsifying Operator

SISSO-Py 是 SISSO（Sure Independence Screening and Sparsifying Operator）算法的纯 Python 实现。SISSO 是一种用于符号回归和特征发现的机器学习方法，特别适用于从有限数据中发现简洁且物理意义明确的数学公式。

## 🌟 主要特性

- **符号特征生成**：通过组合基础操作符生成复杂的符号特征
- **分层复杂度控制**：基于 K 层架构，精确控制特征复杂度
- **多种筛选方法**：支持皮尔逊相关性、互信息等特征筛选策略
- **稀疏建模**：集成 OMP、Lasso、ElasticNet 等稀疏回归方法
- **物理量纲检查**：可选的量纲一致性验证，确保物理意义
- **丰富的操作符库**：包含代数、幂函数、对数、三角函数、双曲函数等
- **自定义函数支持**：支持用户定义的自定义操作符
- **全面的日志系统**：详细的运行过程记录和结果报告

## 📦 安装

### 依赖要求

```python
# 必需依赖
numpy >= 1.19.0
scipy >= 1.7.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
tqdm >= 4.60.0

# 可选依赖
sympy >= 1.9.0  # 用于符号表达式处理
joblib >= 1.1.0  # 用于并行计算
numba >= 0.56.0  # 用于性能加速
```

### 🔧 开发模式安装指南

#### 方法1：标准开发模式安装（推荐）

```bash
# 1. 克隆或进入项目目录
git clone https://github.com/lishuzheng01/sisso-py.git
cd sisso-py

# 2. 创建虚拟环境（可选但推荐）
python -m venv sisso_dev
# Windows PowerShell:
sisso_dev\Scripts\Activate.ps1
# Windows CMD:
# sisso_dev\Scripts\activate.bat
# Linux/macOS:
# source sisso_dev/bin/activate

# 3. 升级基础工具
python -m pip install --upgrade pip setuptools wheel

# 4. 开发模式安装
pip install -e .

# 5. 安装开发依赖（可选）
pip install -r requirements-dev.txt

# 6. 安装完整功能（可选）
pip install -e .[full]
```

#### 方法2：一键安装所有功能

```bash
# 安装包含所有功能的开发版本
pip install -e .[full,dev]
```

#### 方法3：使用现代 Python 工具链

```bash
# 安装 build 工具
pip install build

# 构建项目
python -m build

# 开发模式安装
pip install -e .
```

### ✨ 验证安装

```bash
# 测试导入
python -c "from sisso_py import SissoRegressor; print('安装成功！')"

# 检查版本
python -c "import sisso_py; print('版本:', sisso_py.__version__)"

# 运行命令行工具（如果需要）
sisso-py --help
```

### 🔄 开发模式的优势

- **实时更改**：修改代码后无需重新安装，直接生效
- **依赖管理**：自动处理包依赖关系
- **命令行工具**：自动安装 `sisso-py` 命令
- **完整功能**：支持所有导入路径和模块结构

### 📁 项目结构

安装后，您的项目结构应该是这样的：

```
SISSO/
├── sisso_py/                 # 源代码包
│   ├── __init__.py          # 包初始化（含版本号）
│   ├── config.py            # 全局配置
│   ├── ops/                 # 操作符模块
│   ├── dsl/                 # 表达式语言
│   ├── gen/                 # 特征生成
│   ├── sis/                 # 筛选和稀疏建模
│   ├── metrics/             # 评估指标
│   ├── model/               # 主要模型
│   ├── io/                  # 输入输出
│   ├── utils/               # 工具函数
│   └── cli.py               # 命令行接口
├── setup.py                 # setuptools 配置
├── pyproject.toml           # 现代项目配置
├── requirements.txt         # 运行依赖
├── requirements-dev.txt     # 开发依赖
├── MANIFEST.in             # 打包清单
├── LICENSE                 # 许可证
├── README.md               # 项目文档
└── sisso计算库技术方案.md    # 技术方案
```

### 🛠️ 开发工作流

1. **修改代码**：直接编辑 `sisso_py/` 目录下的文件
2. **测试修改**：

   ```bash
   python -c "from sisso_py import SissoRegressor; model = SissoRegressor()"
   ```

3. **运行测试**：

   ```bash
   pytest tests/  # 如果有测试文件
   ```

4. **代码格式化**：

   ```bash
   black sisso_py/
   isort sisso_py/
   ```

### 🎯 配置文件说明

项目包含以下配置文件：

- **`setup.py`** - 传统的 setuptools 配置
- **`pyproject.toml`** - 现代 Python 项目配置（PEP 518/621 标准）
- **`requirements.txt`** - 运行时依赖
- **`requirements-dev.txt`** - 开发依赖
- **`MANIFEST.in`** - 打包文件清单
- **`LICENSE`** - Apache 2.0 许可证

## 📚 详细使用方法

### 🎯 核心概念

在使用 SISSO-Py 之前，了解以下核心概念将帮助您更好地配置和使用该库：

#### 1. 符号回归流程

SISSO 算法遵循以下流程：

1. **特征生成**：从原始特征出发，通过操作符组合生成新特征
2. **分层扩展**：按复杂度层次（K层）逐步扩展特征空间
3. **特征筛选（SIS）**：使用统计方法筛选最相关的特征
4. **稀疏建模（SO）**：使用稀疏回归构建最终模型

#### 2. 关键参数说明

- **K**：最大复杂度层数，控制特征生成的深度
- **operators**：可用的数学操作符集合
- **sis_screener**：特征筛选方法（'pearson'、'mutual_info'）
- **sis_topk**：每层保留的特征数量
- **so_solver**：稀疏求解器类型（'omp'、'lasso'、'elasticnet'）
- **so_max_terms**：最终模型的最大项数

### 🔧 基础配置与初始化

#### 最简配置

```python
from sisso_py import SissoRegressor

# 使用默认参数
model = SissoRegressor()
```

#### 自定义配置

```python
model = SissoRegressor(
    K=3,                              # 最大复杂度层数
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square'],
    sis_screener='pearson',           # 特征筛选方法
    sis_topk=1000,                   # SIS保留特征数
    so_solver='omp',                 # 稀疏求解器
    so_max_terms=5,                  # 最终模型最大项数
    cv=5,                            # 交叉验证折数
    random_state=42,                 # 随机种子
    n_jobs=-1                        # 并行作业数
)
```

### 📊 数据准备与处理

#### 从 Pandas DataFrame 准备数据

```python
import pandas as pd
import numpy as np

# 创建示例数据
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'temperature': np.random.uniform(200, 400, n_samples),    # 温度
    'pressure': np.random.uniform(1, 10, n_samples),         # 压强
    'volume': np.random.uniform(0.1, 2, n_samples),          # 体积
})

# 模拟理想气体定律: PV = nRT (简化为 P = T/V)
data['target'] = data['temperature'] / data['volume'] + np.random.normal(0, 0.1, n_samples)

# 分离特征和目标
X = data[['temperature', 'pressure', 'volume']]
y = data['target']
```

#### 从 NumPy 数组准备数据

```python
from sisso_py.io import load_from_numpy

# NumPy 数组数据
X_np = np.random.randn(100, 4)
y_np = X_np[:, 0]**2 + X_np[:, 1] - 0.5 * X_np[:, 2] * X_np[:, 3]

# 转换为 DataFrame
X, y = load_from_numpy(X_np, y_np, feature_names=['x1', 'x2', 'x3', 'x4'])
```

#### 处理实际数据文件

```python
from sisso_py.io import load_from_pandas

# 从 CSV 文件加载
df = pd.read_csv('your_data.csv')
X, y = load_from_pandas(df, target_column='target_variable')

# 数据预处理（可选）
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X), 
    columns=X.columns, 
    index=X.index
)
```

### 🎛️ 操作符配置详解

#### 内置操作符完整列表

```python
# 基础代数运算
basic_ops = ['+', '-', '*', 'safe_div']

# 幂和根运算
power_ops = ['sqrt', 'cbrt', 'square', 'poly2', 'poly3']

# 对数和指数
log_exp_ops = ['log', 'log10', 'exp']

# 绝对值和符号
abs_sign_ops = ['abs', 'sign']

# 三角函数
trig_ops = ['sin', 'cos']

# 双曲函数
hyperbolic_ops = ['sinh', 'cosh', 'tanh']

# 物理相关
physics_ops = ['reciprocal']

# 组合使用
all_operators = basic_ops + power_ops + log_exp_ops + abs_sign_ops + trig_ops + hyperbolic_ops + physics_ops

model = SissoRegressor(operators=all_operators)
```

#### 自定义操作符详细示例

```python
import numpy as np

# 定义自定义函数
def exp_decay(x, decay_rate=0.1):
    """指数衰减函数"""
    return np.exp(-decay_rate * np.abs(x))

def polynomial_custom(x, degree=3):
    """自定义多项式"""
    return x ** degree

def combined_function(x, y):
    """组合函数"""
    return np.sqrt(x**2 + y**2)

# 方式1：直接传入（使用默认设置）
model = SissoRegressor(
    operators=[
        '+', '-', '*', 'safe_div',
        exp_decay,           # 默认名称: 'exp_decay', 复杂度: 2
        polynomial_custom,   # 默认名称: 'polynomial_custom', 复杂度: 2
        combined_function    # 默认名称: 'combined_function', 复杂度: 2
    ]
)

# 方式2：详细配置
model = SissoRegressor(
    operators=[
        '+', '-', '*', 'safe_div',
        (exp_decay, {
            'name': 'exp_decay',
            'complexity_cost': 3,
            'validity_checker': lambda x: np.isfinite(x)  # 可选：有效性检查
        }),
        (polynomial_custom, {
            'name': 'poly_custom',
            'complexity_cost': 4
        }),
        (combined_function, {
            'name': 'norm2d',
            'complexity_cost': 3
        })
    ]
)
```

### 🔍 特征筛选策略详解

#### 皮尔逊相关性筛选

```python
model_pearson = SissoRegressor(
    sis_screener='pearson',
    sis_topk=1000,               # 保留相关性最高的1000个特征
    K=3
)
```

#### 互信息筛选

```python
model_mi = SissoRegressor(
    sis_screener='mutual_info',
    sis_topk=800,                # 互信息通常更严格，可以用较少的特征数
    K=3
)
```

#### 动态筛选策略

```python
# 根据数据规模调整筛选参数
n_samples, n_features = X.shape

if n_samples < 100:
    topk = min(500, n_samples * 10)
elif n_samples < 1000:
    topk = min(2000, n_samples * 5)
else:
    topk = min(5000, n_samples * 2)

model = SissoRegressor(
    sis_screener='pearson',
    sis_topk=topk,
    K=min(4, int(np.log2(n_features)) + 1)  # 根据特征数调整复杂度
)
```

### 🎯 稀疏建模配置详解

#### OMP (Orthogonal Matching Pursuit)

```python
model_omp = SissoRegressor(
    so_solver='omp',
    so_max_terms=3,              # 最多选择3个特征
    cv=5                         # 5折交叉验证
)
```

#### Lasso 回归

```python
model_lasso = SissoRegressor(
    so_solver='lasso',
    cv=10,                       # 更多的交叉验证折数
    # Lasso 会自动根据正则化参数选择特征数
)
```

#### ElasticNet 回归

```python
model_en = SissoRegressor(
    so_solver='elasticnet',
    cv=5,
    # ElasticNet 结合了 L1 和 L2 正则化
)
```

#### 高级稀疏建模配置

```python
# 使用交叉验证自动选择最佳参数
from sklearn.model_selection import TimeSeriesSplit

model = SissoRegressor(
    so_solver='lasso',
    cv=TimeSeriesSplit(n_splits=5),  # 时间序列交叉验证
)
```

### 🏃‍♂️ 模型训练与预测流程

#### 基础训练流程

```python
# 1. 创建模型
model = SissoRegressor(
    K=3,
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square'],
    sis_screener='pearson',
    sis_topk=1000,
    so_solver='omp',
    so_max_terms=5,
    cv=5,
    random_state=42
)

# 2. 训练模型
print("开始训练...")
model.fit(X, y)
print("训练完成！")

# 3. 进行预测
y_pred = model.predict(X)

# 4. 评估性能
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
```

#### 训练验证分离

```python
from sklearn.model_selection import train_test_split

# 分离训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 在训练集上训练
model.fit(X_train, y_train)

# 在测试集上评估
y_pred_test = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
```

### 📋 结果分析与解释

#### 获取详细报告

```python
# 获取完整报告
report = model.explain()

# 打印模型配置
print("=== 模型配置 ===")
config = report['configuration']
for key, value in config.items():
    print(f"{key}: {value}")

# 打印最终公式
print("\n=== 发现的公式 ===")
final_model = report['results']['final_model']
print(f"LaTeX 格式: {final_model['formula_latex']}")
print(f"SymPy 格式: {final_model['formula_sympy']}")
print(f"截距: {final_model['intercept']:.4f}")

# 打印特征信息
print("\n=== 选中的特征 ===")
for i, feature in enumerate(final_model['features'], 1):
    print(f"{i}. {feature['signature']}")
    print(f"   系数: {feature['coefficient']:.4f}")
    print(f"   复杂度: {feature['complexity']}")
    print(f"   LaTeX: {feature['latex']}")
    print()

# 打印运行统计
print("=== 运行统计 ===")
run_info = report['run_info']
print(f"生成特征总数: {run_info['total_features_generated']}")
print(f"SIS后特征数: {run_info['features_after_sis']}")
print(f"最终模型特征数: {run_info['features_in_final_model']}")
```

#### 导出结果

```python
from sisso_py.io import export_to_latex, export_to_sympy, export_to_json

# 导出 LaTeX 公式
latex_formula = export_to_latex(model)
print("LaTeX 公式:", latex_formula)

# 导出 SymPy 表达式
sympy_expr = export_to_sympy(model)
print("SymPy 表达式:", sympy_expr)

# 导出完整报告
export_to_json(model, "sisso_results.json")
print("完整报告已保存到 sisso_results.json")

# 保存模型（使用 pickle）
import pickle
with open('sisso_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("模型已保存到 sisso_model.pkl")
```

### 🔬 物理量纲一致性检查

#### 量纲系统说明

SISSO-Py 支持基于国际单位制（SI）的7个基本量纲：

- **M**: 质量 (Mass)
- **L**: 长度 (Length)
- **T**: 时间 (Time)
- **I**: 电流 (Electric Current)
- **Θ**: 温度 (Temperature)
- **N**: 物质的量 (Amount of Substance)
- **J**: 发光强度 (Luminous Intensity)

#### 常见量纲定义

```python
from sisso_py.dsl.dimension import Dimension

# 基本量纲
mass_dim = Dimension([1, 0, 0, 0, 0, 0, 0])        # M
length_dim = Dimension([0, 1, 0, 0, 0, 0, 0])      # L
time_dim = Dimension([0, 0, 1, 0, 0, 0, 0])        # T
current_dim = Dimension([0, 0, 0, 1, 0, 0, 0])     # I
temperature_dim = Dimension([0, 0, 0, 0, 1, 0, 0]) # Θ

# 复合量纲
velocity_dim = Dimension([0, 1, -1, 0, 0, 0, 0])   # L/T
acceleration_dim = Dimension([0, 1, -2, 0, 0, 0, 0]) # L/T²
force_dim = Dimension([1, 1, -2, 0, 0, 0, 0])      # MLT⁻² (牛顿)
energy_dim = Dimension([1, 2, -2, 0, 0, 0, 0])     # ML²T⁻² (焦耳)
power_dim = Dimension([1, 2, -3, 0, 0, 0, 0])      # ML²T⁻³ (瓦特)
pressure_dim = Dimension([1, -1, -2, 0, 0, 0, 0])  # ML⁻¹T⁻² (帕斯卡)

# 无量纲
dimensionless = Dimension([0, 0, 0, 0, 0, 0, 0])
```

#### 带量纲检查的完整示例

```python
import numpy as np
import pandas as pd
from sisso_py import SissoRegressor
from sisso_py.dsl.dimension import Dimension

# 模拟理想气体数据
np.random.seed(42)
n_samples = 100

# 生成数据
temperature = np.random.uniform(250, 350, n_samples)  # K
volume = np.random.uniform(0.1, 1.0, n_samples)      # m³
n_moles = np.random.uniform(0.5, 2.0, n_samples)     # mol

# 理想气体定律: PV = nRT => P = nRT/V
R = 8.314  # 气体常数
pressure = (n_moles * R * temperature / volume) + np.random.normal(0, 100, n_samples)

# 创建数据框
data = pd.DataFrame({
    'temperature': temperature,
    'volume': volume,
    'moles': n_moles,
    'pressure': pressure
})

# 定义量纲
temp_dim = Dimension([0, 0, 0, 0, 1, 0, 0])        # Θ (温度)
volume_dim = Dimension([0, 3, 0, 0, 0, 0, 0])      # L³ (体积)
moles_dim = Dimension([0, 0, 0, 0, 0, 1, 0])       # N (物质的量)
pressure_dim = Dimension([1, -1, -2, 0, 0, 0, 0])  # ML⁻¹T⁻² (压强)

feature_dimensions = {
    'temperature': temp_dim,
    'volume': volume_dim,
    'moles': moles_dim
}

# 创建模型（启用量纲检查）
model = SissoRegressor(
    K=3,
    operators=['+', '-', '*', 'safe_div', 'reciprocal'],
    dimensional_check=True,
    sis_topk=500,
    so_max_terms=3,
    random_state=42
)

# 准备数据
X = data[['temperature', 'volume', 'moles']]
y = data['pressure']

# 训练（传入量纲信息）
model.fit(X, y, 
          feature_dimensions=feature_dimensions,
          target_dimension=pressure_dim)

# 查看结果
report = model.explain()
print("发现的公式（应该类似 PV = nRT）:")
print(report['results']['final_model']['formula_latex'])
```

### 🔥 高级使用技巧

#### 多目标并行训练

```python
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def train_sisso_model(config):
    """训练单个SISSO模型的函数"""
    X, y, params = config
    model = SissoRegressor(**params)
    model.fit(X, y)
    return model.explain()

# 定义多个配置
configs = [
    (X, y, {'K': 2, 'so_solver': 'omp', 'so_max_terms': 3}),
    (X, y, {'K': 3, 'so_solver': 'lasso'}),
    (X, y, {'K': 2, 'so_solver': 'elasticnet'}),
]

# 并行训练
with ProcessPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(train_sisso_model, configs))

# 比较结果
for i, result in enumerate(results):
    print(f"配置 {i+1} 的最佳公式:")
    print(result['results']['final_model']['formula_latex'])
    print()
```

#### 自适应参数调整

```python
def adaptive_sisso_training(X, y, max_complexity=5):
    """自适应调整SISSO参数进行训练"""
    n_samples, n_features = X.shape
    
    best_model = None
    best_score = float('-inf')
    
    for K in range(2, max_complexity + 1):
        # 根据数据规模调整参数
        if n_samples < 100:
            sis_topk = min(200, n_samples * 3)
            so_max_terms = min(3, n_features)
        elif n_samples < 500:
            sis_topk = min(1000, n_samples * 2)
            so_max_terms = min(5, n_features)
        else:
            sis_topk = min(2000, n_samples)
            so_max_terms = min(8, n_features)
        
        # 训练模型
        model = SissoRegressor(
            K=K,
            sis_topk=sis_topk,
            so_max_terms=so_max_terms,
            cv=5,
            random_state=42
        )
        
        try:
            model.fit(X, y)
            
            # 计算交叉验证得分
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer, r2_score
            
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

# 使用自适应训练
best_model, best_score = adaptive_sisso_training(X, y)
print(f"\n最佳模型 R²: {best_score:.4f}")
print("最佳公式:", best_model.explain()['results']['final_model']['formula_latex'])
```

#### 集成学习方法

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 创建多个SISSO模型
sisso1 = SissoRegressor(K=2, so_solver='omp', random_state=42)
sisso2 = SissoRegressor(K=3, so_solver='lasso', random_state=43)
sisso3 = SissoRegressor(K=2, so_solver='elasticnet', random_state=44)

# 创建其他基学习器
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()

# 创建投票回归器
ensemble = VotingRegressor([
    ('sisso_omp', sisso1),
    ('sisso_lasso', sisso2),
    ('sisso_en', sisso3),
    ('rf', rf),
    ('lr', lr)
])

# 训练集成模型
ensemble.fit(X, y)
y_pred_ensemble = ensemble.predict(X)

# 评估集成效果
from sklearn.metrics import r2_score
ensemble_r2 = r2_score(y, y_pred_ensemble)
print(f"集成模型 R²: {ensemble_r2:.4f}")
```

#### 时间序列数据处理

```python
from sklearn.model_selection import TimeSeriesSplit

# 时间序列数据示例
np.random.seed(42)
n_points = 500
time = np.linspace(0, 10, n_points)

# 生成带趋势和季节性的时间序列
trend = 0.1 * time
seasonal = 2 * np.sin(2 * np.pi * time)
noise = np.random.normal(0, 0.2, n_points)
ts_data = trend + seasonal + noise

# 创建滞后特征
def create_lag_features(data, max_lag=5):
    """创建滞后特征"""
    df = pd.DataFrame({'y': data})
    
    for lag in range(1, max_lag + 1):
        df[f'y_lag_{lag}'] = df['y'].shift(lag)
    
    # 添加时间特征
    df['time'] = range(len(data))
    df['time_squared'] = df['time'] ** 2
    
    return df.dropna()

# 准备时间序列数据
ts_df = create_lag_features(ts_data, max_lag=3)
X_ts = ts_df.drop('y', axis=1)
y_ts = ts_df['y']

# 使用时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

model_ts = SissoRegressor(
    K=3,
    operators=['+', '-', '*', 'safe_div', 'sin', 'cos'],
    cv=tscv,  # 使用时间序列交叉验证
    sis_topk=500,
    so_max_terms=5
)

model_ts.fit(X_ts, y_ts)

# 查看时间序列模型结果
ts_report = model_ts.explain()
print("时间序列模型公式:")
print(ts_report['results']['final_model']['formula_latex'])
```

### 💡 最佳实践和性能优化

#### 数据预处理建议

```python
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def preprocess_data(X, y, method='standard'):
    """数据预处理函数"""
    
    # 1. 处理缺失值
    X_clean = X.fillna(X.median())
    
    # 2. 移除常量列
    constant_cols = X_clean.columns[X_clean.nunique() <= 1]
    if len(constant_cols) > 0:
        print(f"移除常量列: {list(constant_cols)}")
        X_clean = X_clean.drop(constant_cols, axis=1)
    
    # 3. 数据缩放
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()  # 对异常值更鲁棒
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None
    
    if scaler:
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
    else:
        X_scaled = X_clean
    
    # 4. 移除高相关特征
    corr_matrix = X_scaled.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_pairs = [
        column for column in upper_tri.columns 
        if any(upper_tri[column] > 0.95)
    ]
    
    if high_corr_pairs:
        print(f"移除高相关特征: {high_corr_pairs}")
        X_scaled = X_scaled.drop(high_corr_pairs, axis=1)
    
    return X_scaled, y, scaler

# 使用预处理
X_processed, y_processed, scaler = preprocess_data(X, y, method='robust')
```

#### 内存优化策略

```python
def memory_efficient_sisso(X, y, batch_size=1000):
    """内存优化的SISSO训练"""
    
    n_samples = len(X)
    
    if n_samples > 10000:
        # 大数据集：分批处理
        print("检测到大数据集，使用分批处理...")
        
        # 先用小样本训练获得特征集
        sample_idx = np.random.choice(n_samples, min(5000, n_samples), replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        # 训练小模型获取重要特征
        small_model = SissoRegressor(
            K=2,
            sis_topk=500,
            so_max_terms=3,
            random_state=42
        )
        small_model.fit(X_sample, y_sample)
        
        # 在全数据集上使用发现的特征进行最终训练
        important_features = [
            f['signature'] for f in 
            small_model.explain()['results']['final_model']['features']
        ]
        
        print(f"发现重要特征: {important_features}")
        
        # 使用更保守的参数在全数据集上训练
        final_model = SissoRegressor(
            K=3,
            sis_topk=1000,
            so_max_terms=5,
            random_state=42
        )
        final_model.fit(X, y)
        
        return final_model
    
    else:
        # 小数据集：正常处理
        model = SissoRegressor(
            K=3,
            sis_topk=min(2000, len(X) * 5),
            so_max_terms=min(8, X.shape[1]),
            random_state=42
        )
        model.fit(X, y)
        return model

# 使用内存优化训练
optimized_model = memory_efficient_sisso(X, y)
```

#### 结果验证和鲁棒性检查

```python
def validate_sisso_results(model, X, y, n_bootstrap=100):
    """验证SISSO结果的鲁棒性"""
    
    from sklearn.utils import resample
    from sklearn.metrics import mean_squared_error, r2_score
    
    results = {
        'formulas': [],
        'r2_scores': [],
        'rmse_scores': [],
        'coefficients': []
    }
    
    print("进行Bootstrap验证...")
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap Sampling"):
        # 重采样
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # 训练模型
        try:
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
            boot_model.fit(X_boot, y_boot)
            
            # 评估
            y_pred = boot_model.predict(X_boot)
            r2 = r2_score(y_boot, y_pred)
            rmse = np.sqrt(mean_squared_error(y_boot, y_pred))
            
            # 记录结果
            report = boot_model.explain()
            formula = report['results']['final_model']['formula_latex']
            coeffs = [f['coefficient'] for f in report['results']['final_model']['features']]
            
            results['formulas'].append(formula)
            results['r2_scores'].append(r2)
            results['rmse_scores'].append(rmse)
            results['coefficients'].append(coeffs)
            
        except Exception as e:
            continue
    
    # 分析结果
    print(f"\n=== Bootstrap验证结果 (n={len(results['r2_scores'])}) ===")
    print(f"R² 平均值: {np.mean(results['r2_scores']):.4f} ± {np.std(results['r2_scores']):.4f}")
    print(f"RMSE 平均值: {np.mean(results['rmse_scores']):.4f} ± {np.std(results['rmse_scores']):.4f}")
    
    # 统计公式出现频率
    from collections import Counter
    formula_counts = Counter(results['formulas'])
    print(f"\n最常见的公式 (出现次数):")
    for formula, count in formula_counts.most_common(5):
        print(f"  {count:3d}x: {formula}")
    
    return results

# 进行鲁棒性验证
validation_results = validate_sisso_results(model, X, y, n_bootstrap=50)
```

### 🚀 快速开始

### 基础用法

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
    K=3,  # 最大复杂度层数
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log'],
    sis_screener='pearson',  # 筛选方法
    sis_topk=1000,  # SIS 保留特征数
    so_solver='omp',  # 稀疏求解器
    so_max_terms=5,  # 最终模型最大项数
    cv=5  # 交叉验证折数
)

# 拟合模型
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 获取解释和报告
report = model.explain()
print("最佳公式:", report['results']['final_model']['formula_latex'])
print("模型系数:", report['results']['final_model']['features'])
```

### 带物理量纲检查的用法

```python
from sisso_py import SissoRegressor
from sisso_py.dsl.dimension import Dimension

# 定义物理量纲 (质量M, 长度L, 时间T, 电流I, 温度Θ, 物质量N, 发光强度J)
length_dim = Dimension([0, 1, 0, 0, 0, 0, 0])  # 长度 L
time_dim = Dimension([0, 0, 1, 0, 0, 0, 0])    # 时间 T
velocity_dim = Dimension([0, 1, -1, 0, 0, 0, 0])  # 速度 L/T

# 准备带量纲的数据
feature_dimensions = {
    'distance': length_dim,
    'time': time_dim,
    'initial_velocity': velocity_dim
}

model = SissoRegressor(
    K=2,
    dimensional_check=True,  # 启用量纲检查
    operators=['+', '-', '*', 'safe_div', 'square']
)

# 拟合时传入量纲信息
model.fit(X, y, 
          feature_dimensions=feature_dimensions,
          target_dimension=length_dim)  # 目标是长度量纲
```

### 使用自定义操作符

```python
import numpy as np

def gaussian(x):
    """高斯函数"""
    return np.exp(-x**2)

def sigmoid(x):
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-x))

# 方式1: 直接传入函数
model = SissoRegressor(
    K=2,
    operators=[
        '+', '-', '*', 'safe_div',
        gaussian,  # 直接传入函数
        sigmoid
    ]
)

# 方式2: 传入函数和配置的元组
model = SissoRegressor(
    K=2,
    operators=[
        '+', '-', '*', 'safe_div',
        (gaussian, {'name': 'gauss', 'complexity_cost': 3}),
        (sigmoid, {'name': 'sig', 'complexity_cost': 4})
    ]
)
```

## 📊 完整示例

### 例1：Kepler 第三定律发现

```python
import numpy as np
import pandas as pd
from sisso_py import SissoRegressor
from sisso_py.dsl.dimension import Dimension

# 生成开普勒定律数据: T^2 ∝ a^3
np.random.seed(42)
n_samples = 50

# 半长轴 (天文单位)
a = np.random.uniform(0.5, 5.0, n_samples)
# 周期 (年)，加入噪声
T = np.sqrt(a**3) + np.random.normal(0, 0.05, n_samples)

# 创建数据框
data = pd.DataFrame({
    'semi_major_axis': a,
    'orbital_period': T
})

# 定义量纲
length_dim = Dimension([0, 1, 0, 0, 0, 0, 0])  # L
time_dim = Dimension([0, 0, 1, 0, 0, 0, 0])    # T

feature_dims = {'semi_major_axis': length_dim}
target_dim = time_dim

# 训练模型
model = SissoRegressor(
    K=3,
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'poly3'],
    dimensional_check=True,
    sis_topk=500,
    so_max_terms=3
)

X = data[['semi_major_axis']]
y = data['orbital_period']

model.fit(X, y, feature_dimensions=feature_dims, target_dimension=target_dim)

# 查看结果
report = model.explain()
print("发现的公式:", report['results']['final_model']['formula_latex'])
print("复杂度:", report['run_info']['features_in_final_model'])
```

### 例2：数据驱动的物理公式发现

```python
import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

# 模拟数据：能量公式 E = 1/2 * m * v^2
np.random.seed(123)
n = 200

mass = np.random.uniform(1, 10, n)
velocity = np.random.uniform(0, 20, n)
energy = 0.5 * mass * velocity**2 + np.random.normal(0, 0.1, n)

data = pd.DataFrame({
    'mass': mass,
    'velocity': velocity,
    'energy': energy
})

# 不使用量纲检查的简单模式
model = SissoRegressor(
    K=3,
    operators=['+', '-', '*', 'safe_div', 'square', 'sqrt'],
    sis_screener='mutual_info',  # 使用互信息筛选
    sis_topk=1000,
    so_solver='lasso',  # 使用 Lasso 求解器
    so_max_terms=3,
    cv=10
)

X = data[['mass', 'velocity']]
y = data['energy']

model.fit(X, y)

# 获取详细报告
report = model.explain()
print("模型配置:", report['configuration'])
print("最终公式:", report['results']['final_model']['formula_latex'])
print("特征信息:")
for feature in report['results']['final_model']['features']:
    print(f"  {feature['signature']}: 系数={feature['coefficient']:.4f}, 复杂度={feature['complexity']}")

print(f"\n生成特征总数: {report['run_info']['total_features_generated']}")
print(f"SIS后特征数: {report['run_info']['features_after_sis']}")
print(f"最终模型特征数: {report['run_info']['features_in_final_model']}")
```

## 🔧 高级配置

### 操作符配置

```python
# 内置操作符
AVAILABLE_OPERATORS = [
    # 基础代数
    '+', '-', '*', 'safe_div',
    
    # 幂和根
    'sqrt', 'cbrt', 'square', 'poly2', 'poly3',
    
    # 对数和指数
    'log', 'log10', 'exp',
    
    # 绝对值和符号
    'abs', 'sign',
    
    # 三角函数
    'sin', 'cos',
    
    # 双曲函数
    'sinh', 'cosh', 'tanh',
    
    # 物理相关
    'reciprocal'
]

# 自定义复杂度权重
model = SissoRegressor(
    operators=[
        '+',     # 复杂度 1
        '*',     # 复杂度 1  
        'sqrt',  # 复杂度 2
        'log',   # 复杂度 2
        'sin',   # 复杂度 3
    ]
)
```

### 筛选策略配置

```python
# 不同的特征筛选方法
model_pearson = SissoRegressor(sis_screener='pearson')      # 皮尔逊相关系数
model_mi = SissoRegressor(sis_screener='mutual_info')       # 互信息

# 筛选参数调整
model = SissoRegressor(
    sis_topk=2000,      # 每层保留的特征数
    K=4,                # 最大层数
    so_max_terms=5      # 最终模型最大项数
)
```

### 稀疏求解器配置

```python
# OMP (Orthogonal Matching Pursuit)
model_omp = SissoRegressor(
    so_solver='omp',
    so_max_terms=3
)

# Lasso
model_lasso = SissoRegressor(
    so_solver='lasso',
    # Lasso 会根据 alpha 参数自动确定特征数
)

# ElasticNet
model_en = SissoRegressor(
    so_solver='elasticnet',
    # 结合 L1 和 L2 正则化
)
```

## 📈 结果导出和可视化

### 导出公式

```python
from sisso_py.io import export_to_latex, export_to_sympy, export_to_json

# 导出为 LaTeX
latex_formula = export_to_latex(model)
print("LaTeX 公式:", latex_formula)

# 导出为 SymPy 对象
sympy_expr = export_to_sympy(model)
print("SymPy 表达式:", sympy_expr)

# 导出完整报告为 JSON
export_to_json(model, "sisso_report.json")
```

### 数据接口

```python
from sisso_py.io import load_from_pandas, load_from_numpy

# 从 pandas DataFrame 加载
df = pd.read_csv("data.csv")
X, y = load_from_pandas(df, target_column='target')

# 从 NumPy 数组加载
X_np = np.random.randn(100, 3)
y_np = np.random.randn(100)
X, y = load_from_numpy(X_np, y_np, feature_names=['f1', 'f2', 'f3'])
```

## ⚙️ 性能优化

### 并行计算

```python
model = SissoRegressor(
    n_jobs=-1,  # 使用所有可用 CPU 核心
    # n_jobs=4   # 或指定具体核心数
)
```

### 内存和复杂度控制

```python
model = SissoRegressor(
    K=2,                # 降低层数减少特征数量
    sis_topk=500,       # 减少筛选特征数
    so_max_terms=3,     # 限制最终模型复杂度
)
```

## 🐛 故障排除

### 常见问题

1. **内存不足**

   ```python
   # 解决方案：减少特征生成数量
   model = SissoRegressor(K=2, sis_topk=500)
   ```

2. **量纲检查错误**

   ```python
   # 确保量纲定义正确
   length_dim = Dimension([0, 1, 0, 0, 0, 0, 0])  # [M, L, T, I, Θ, N, J]
   ```

3. **收敛问题**

   ```python
   # 尝试不同的求解器
   model = SissoRegressor(so_solver='lasso')  # 或 'elasticnet'
   ```

### 调试模式

```python
from sisso_py.utils.logging import setup_logging
import logging

# 启用详细日志
setup_logging(level=logging.DEBUG)

model = SissoRegressor(...)
model.fit(X, y)  # 将输出详细的调试信息
```

## 📚 API 参考

### SissoRegressor 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| K | int | 2 | 最大复杂度层数 |
| operators | List | DEFAULT_OPERATORS | 使用的操作符列表 |
| sis_screener | str | 'pearson' | 特征筛选方法 |
| sis_topk | int | 2000 | SIS 保留的特征数 |
| so_solver | str | 'omp' | 稀疏求解器类型 |
| so_max_terms | int | 3 | 最终模型最大项数 |
| cv | int | 5 | 交叉验证折数 |
| dimensional_check | bool | False | 是否启用量纲检查 |
| random_state | int | 42 | 随机种子 |
| n_jobs | int | -1 | 并行作业数 |

### 主要方法

- `fit(X, y, feature_dimensions=None, target_dimension=None)`: 训练模型
- `predict(X)`: 进行预测  
- `explain()`: 获取模型解释和报告

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 Apach 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 SISSO 算法的原始作者
- 感谢 scikit-learn 社区提供的优秀机器学习框架
- 感谢所有贡献者和用户的支持

## 📞 联系方式

- 项目主页: <https://github.com/lishuzheng01/sisso-py>
- 问题报告: <https://github.com/lishuzheng01/sisso-py/issues>
- 邮箱: <3035326878@qq.com>

---

**SISSO-Py**: 让符号回归和公式发现变得简单! 🚀
