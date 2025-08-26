# AllRight-SR: 多策略符号回归工具箱

## 项目说明
AllRight-SR 集成了进化、稀疏建模、贝叶斯、神经符号以及多种混合优化方法，
帮助用户从数据中自动发现具有物理意义的解析表达式。
所有模型在训练后均可通过统一的 `model.explain()` 接口获取评估指标和人类可读的拟合公式。

## 安装
```bash
pip install -r requirements.txt
# 或从源代码安装
pip install -e .
```

## 快速开始
以 SISSO 基础模型为例：
```python
import numpy as np
from sisso_py.sparse_regression.sisso import SISSORegressor

X = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
y = np.sin(X).ravel()

model = SISSORegressor(K=2)
model.fit(X, y)
print(model.explain())
```
典型的报告内容如下所示：
```
y = 0.99*sin(x)
R^2: 0.98
MSE: 0.01
RMSE: 0.10
MAE: 0.08
```

## 算法示例
以下示例默认已准备好训练数据 `X` 与目标 `y`。
每个模型训练完成后均可调用 `model.explain()` 输出包含 R²、MSE 等指标的整洁报告。

### 进化算法
```python
from sisso_py.evolutionary.gp import GeneticProgramming
model = GeneticProgramming(population_size=50, n_generations=10)
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.evolutionary.ga_pso import GAPSORegressor
model = GAPSORegressor(generations=30)
model.fit(X, y)
print(model.explain())
```

### 稀疏建模
```python
from sisso_py.sparse_regression.sisso import SISSORegressor
model = SISSORegressor(K=2)
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor, RidgeRegressor, OMPRegressor
model = LassoRegressor()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.sparse_regression.sindy import SINDyRegressor
model = SINDyRegressor()
model.fit(X, y)
print(model.explain())
```

### 贝叶斯与概率方法
```python
from sisso_py.probabilistic.bsr import BayesianSymbolicRegressor
model = BayesianSymbolicRegressor()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.probabilistic.ppi import ProbabilisticProgramInduction
model = ProbabilisticProgramInduction()
model.fit(X, y)
print(model.explain())
```

### 神经符号
```python
from sisso_py.neural_symbolic.deep_sr import DeepSymbolicRegression
model = DeepSymbolicRegression()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
model = NeuralSymbolicHybrid()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
model = ReinforcementSymbolicRegression()
model.fit(X, y)
print(model.explain())
```

### 混合优化
```python
from sisso_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
model = EvolutionaryGradientHybrid()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
model = MultiObjectiveSymbolicRegression()
model.fit(X, y)
print(model.explain())
```
```python
from sisso_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
model = PhysicsInformedSymbolicRegression()
model.fit(X, y)
print(model.explain())
```

## 贡献
欢迎提交 Issue 或 Pull Request 来改进项目。

## 许可证
本项目遵循 Apache 许可证，详情参见 [LICENSE](LICENSE)。

## 联系
如有问题或建议，请在 GitHub Issues 中反馈或联系 3035326878@qq.com。

---
**AllRight-SR**：让符号回归和公式发现更加简单。
