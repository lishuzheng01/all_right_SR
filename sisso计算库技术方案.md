

# 一、总体目标与边界

* 目标：实现 SISSO（Sure Independence Screening and Sparsifying Operator）流程：**符号特征生成 → SIS 筛选 → 稀疏建模（SO） → 模型与公式输出**。
* 约束：仅依赖 Python 科学计算生态；不依赖 C/CUDA 扩展。性能通过 **NumPy 向量化 + Numba（可选 JIT）+ Joblib 并行**保障。
* 产物：可安装的 `sisso_py/` 包、命令行入口（可选）、统一日志与评估输出。

# 二、包结构设计

```
sisso_py/
  sisso_py
  config.py                # 全局配置、随机种子、默认操作符集
  ops/
    __init__.py
    base.py                # Operator 抽象、注册器
    algebra.py             # 基本代数
    power_root.py          # 幂与开方
    log_exp.py             # 对数与指数
    abs_sign.py            # 绝对值与符号函数
    poly.py                # 多项式扩展
    physics.py             # 常见物理算子

  dsl/
    expr.py                # 表达式树(节点/常量/变量/一元/二元)
    complexity.py          # 复杂度度量与K层控制
    dimension.py           # （可选）量纲检查
  gen/
    generator.py           # 表达式空间生成/去重/裁剪
    evaluator.py           # 批量数值求值（NumPy/Numba）
    hashing.py             # 表达式规范化、hash 去重
  sis/
    screening.py           # SIS：相关性/互信息/HSIC等筛选
    so.py                  # SO：稀疏建模(L0近似/OMP/Lasso/ElasticNet)
  metrics/
    regression.py          # RMSE/MAE/R2/跨验证
    symbolic.py            # 公式复杂度、稳定性、可解释性
  model/
    pipeline.py            # SissoRegressor 主类（fit/predict/explain）
    report.py              # 文本/JSON 报告、最佳公式与指标
  io/
    dataset.py             # 数据接口（numpy/pandas）
    export.py              # 导出公式(TeX/SymPy/可执行Python)
  utils/
    logging.py             # 日志封装（logging + tqdm）
    parallel.py            # joblib/n_jobs
    seed.py                # 随机种子
  cli.py                   # 可选：命令行入口
```

# 三、核心数据结构

* `Operator`（抽象类）：`name`, `arity`, `callable`, `validity_checker`, `complexity_cost`, `latex_fmt`。
* `Expr`（表达式树）：

  * `Var(name)`, `Const(value)`, `Unary(op, child)`, `Binary(op, left, right)`
  * 方法：`evaluate(X)`, `to_sympy()`, `to_python()`, `to_latex()`, `signature()`（规范化字符串/哈希）
* `ComplexityBudget`：以 **层数 K** 和 **操作符单次代价** 共同约束表达式生成。
* `FeatureSpace`：保存每层生成的表达式、去重映射、其在训练集上的数值向量（懒求值/缓存）。

# 四、符号操作体系（与需求 1 对齐）

1. **基本代数**：`+, -, *, / (safe_div)`

   * 规则：除零保护、NaN/Inf 屏蔽与裁剪（`np.errstate`）。
2. **幂与开方**：`pow(x, n), sqrt | cbrt | abs_pow(x, p)`

   * 规则：根号安全域、奇偶幂化简、对负数开方的处理策略。
3. **对数与指数**：`log, log10, exp`

   * 规则：`log(safe_abs(x)+eps)`，避免非正域错误；`exp` 溢出裁剪。
4. **绝对值与符号函数**：`abs, sign`

   * 规则：`sign(0)=0`，连贯的可导近似（可选）以利于梯度法。
5. **多项式扩展**：`poly_n(x, degree)`、多变量混合项如 `x*y, x^2*y`

   * 规则：复杂度以项次数与变量总数累积。
6. **常见物理算子**（可选集）：`reciprocal`, `sinh/cosh/tanh`, `sin/cos`（若数据有周期性），`norm`（组合特征），`gradient-like`（若有序列/空间索引）。
7. **自定义函数**：通过 `ops.custom.register(name, func, arity, complexity_cost, validity_checker)` 动态注入，并自动纳入 DSL 与生成器。

# 五、复杂度与 K 层控制（与需求 2 对齐）

* **复杂度模型**：

  * 变量/常量代价=1；一元操作代价=1～2；二元操作代价=2～3；多项式高阶按阶数线性/次线性增长（可配置）。
* **层级生成**：

  * L0：原始特征（变量/常量）。
  * L1：在 L0 上施加一层一元/二元操作。
  * …
  * Lk：从 ≤(k-1) 层产物组合生成，且总复杂度不超过 `K`。
* **裁剪策略**：

  * **语义等价去重**（交换律/结合律/幂次合并）、**数值近似等价**（皮尔森相关系数 > 阈值）、**病态筛除**（常量列/全零方差/极端偏态）。

# 六、特征生成与求值引擎

* **生成**：基于算子元数在候选集合上进行笛卡尔组合，使用 **规范化签名 + 哈希** 做结构去重；按复杂度优先/束搜索策略扩展。
* **求值**：NumPy 向量化，必要时用 **Numba JIT** 对热路径（如 `safe_div`, `log1p_abs`）加速；大规模候选采用 **分批+内存映射**。
* **稳定性**：统一 `eps`、`clip_min/max`、`nan_policy='omit'`，保证数值稳定。

# 七、SIS（Sure Independence Screening）

* **打分准则**（可配置）：

  * 皮尔森/斯皮尔曼相关系数的绝对值；
  * **HSIC**（使用 `scipy.spatial.distance` + 高斯核近似）；
  * 互信息（`sklearn.feature_selection.mutual_info_regression`）。
* **流程**：

  1. 对每一层产物按准则排序；
  2. 每层保留 Top-M（或累计保留不超过全局上限 G）；
  3. 与先前层的入围特征合并并去重。
* **并行**：`joblib.Parallel(n_jobs)` 分层或分批计算相关度。

# 八、SO（Sparsifying Operator）稀疏建模

* **候选集合**：来自 SIS 的合并特征矩阵 `Φ ∈ R^{N×P}`。
* **求解器**（可切换）：

  * 近似 L0：**OMP**（`sklearn.linear_model.OrthogonalMatchingPursuit`）；
  * L1：**Lasso/LassoLars**；弹性网：**ElasticNet**；
  * 少量特征的 **子集枚举**（P 较小场景，启发式束搜索：深度优先+早停）。
* **模型选择**：

  * 交叉验证（KFold/GroupKFold/TimeSeriesSplit）；
  * 信息准则（AIC/BIC）与**公式复杂度惩罚**的加权目标；
  * **物理一致性**（可选量纲检查）作为硬约束/软惩罚。

# 九、评价指标与报告（与需求 3 对齐）

* **训练/验证指标**：`RMSE`, `MAE`, `R2`, `MedAE`；
  可选：`SRD`（符号稳定性、对噪声鲁棒性评分）、`Sparsity`（特征数）。
* **最终输出**：

  * 最佳公式（SymPy/LaTeX/Python 可执行字符串）；
  * 系数表与所用原子特征映射；
  * 交叉验证分数摘要、学习曲线（数值表，图形由外部可视工具生成或另包支持）。
* **可复现实验卡**（JSON）：

  * 数据版本、随机种子、操作符集、K、SIS阈值、SO配置与最终指标。

# 十、日志体系（与需求 3.1/3.2 对齐）

* 使用标准 `logging`（INFO/DEBUG）+ `tqdm` 进度条：

  * **运行过程输出**：

    * 配置回显（K、操作符、SIS与SO参数、随机种子）；
    * 每层生成数量、去重前后数量、SIS保留数量；
    * SO 过程中被选入/剔除的特征索引与验证分数变化；
    * 数值异常计数（nan/inf/clip 触发次数）。
  * **最后最佳公式与指标**：

    * 公式（LaTeX 与 Python 形式）、系数、所用原始特征；
    * 训练/验证（或测试）RMSE/MAE/R2、交叉验证均值±方差；
    * 复杂度与可解释性评分。
* 支持将日志同步到 **文件** 和 **stdout**；报告落盘为 `report.json` 与 `best_formula.tex/.py`.



# 十一、API 设计（面向用户，支持直接传入自定义函数）

## 1) 核心类

```python
from sisso_py.model import SissoRegressor

model = SissoRegressor(
    K=2,
    # operators 现在支持三种形式的混用（见下文）
    operators=[
        '+', '-', '*', 'safe_div', 'log', 'exp', 'sqrt', 'abs',
        my_custom_op,                                       # 直接传入def函数
        (my_other_op, {'name': 'my_op', 'complexity_cost': 2, 'validity_checker': None}),
    ],
    sis_score='pearson', sis_topk=2000, global_topk=5000,
    so_solver='omp', so_max_terms=3,
    cv=5, random_state=42, n_jobs=-1, eps=1e-8, clip=1e6,
    dimensional_check=False
)

model.fit(X, y, feature_names=None)  # X: np.ndarray | pd.DataFrame
yhat = model.predict(X_val)
info = model.explain()               # dict: 公式、系数、指标、配置
```

## 2) `operators` 参数的三种可用形态

`operators` 列表允许 **内置字符串**、**可调用函数**、或 **(可调用, 配置字典) 元组** 混合传入：

1. **内置字符串**（如 `'+'`, `'safe_div'`, `'log'` 等）：保持不变。
2. **直接传入 Python 函数**（你用 `def` 写好的可调用）：

   * 我们将自动：

     * 通过 `inspect.signature` 推断**元数（arity）**；
     * 使用 `func.__name__` 作为**算子名**；
     * 赋默认 `complexity_cost=2`、`validity_checker=None`。
   * 如需自定义复杂度或有效性检查，可用第 3 种形式。
3. **(callable, options) 形式**：

   * `options` 可包含：

     * `name: str`（可选，默认 `func.__name__`）
     * `complexity_cost: int`（可选，默认 2）
     * `validity_checker: Callable[..., np.ndarray | bool] | None`（可选）

       * 用于筛除数值非法域（如对数负值、除零等），返回布尔掩码或抛出受控异常。

示例：

```python
def my_custom_op(x, y):
    # 要求：NumPy 向量化友好、可广播；内部自行做安全域处理或交给库的外层裁剪
    return (x - y) / (np.abs(x) + 1.0)

def my_other_op(x):
    return np.log1p(np.abs(x))

model = SissoRegressor(
    K=3,
    operators=[
        '*', 'safe_div', 'sqrt',
        my_custom_op,                                  # 使用默认 name/cost
        (my_other_op, {'name': 'log1p_abs', 'complexity_cost': 1})
    ],
    ...
)
```

## 3) 自定义函数的要求与最佳实践

* **签名与向量化**：仅使用**位置参数**承载输入张量，如 `def f(x)`、`def g(x, y)`；内部使用 **NumPy** 运算、支持广播；避免 Python 级 for 循环。
* **数值稳定性**：建议在函数内部进行基本防护（如 `np.clip`、`np.where`），库也会在外层统一应用 `np.errstate` 与 `nan_policy`。
* **可导性**：本库默认不做梯度优化约束，自定义函数不必可导；如需光滑近似，可自行在函数内实现。
* **性能建议**：尽量用 NumPy 原语；若有热路径，可在后续版本通过 `numba.njit` 自动加速（库会做条件装饰，用户无需改函数）。
* **量纲（可选）**：若启用量纲检查，可在 `operator_configs`（后续扩展）中为自定义函数声明量纲变换；当前版本默认不强制此项。

## 4) 公式导出与可解释性

* 直接传入的函数会在表达式树中以其 `name` 呈现；导出 `SymPy/LaTeX/Python` 时：

  * 若识别到等价 SymPy 表达式，使用对应符号；
  * 否则以 **函数调用形式** 导出（`name(arg1, arg2, ...)`），同时在 `export.py` 中生成该函数的最小可执行包装，保证可复现实验。

## 5) 日志与错误提示

* 在 `fit()` 开始阶段，库会**枚举并打印**所有解析后的算子：`name / arity / complexity_cost / source`（内置或自定义）。
* 若自定义函数的**元数**无法推断（如带 `*args`），或返回值形状与输入不匹配，将在生成阶段抛出 **明确的可读错误**，并标注问题算子名与示例输入形状。



# 十二、性能与内存策略

* **按层分批生成**、**按批求值**（批大小自适应内存）；
* **去重优先**，降低求值次数；
* **Joblib 并行** + **Numba JIT**（可开关）；
* 大型矩阵使用 **内存映射文件**（`numpy.memmap`）；
* 结果缓存（表达式→向量）带 **LRU**（`functools.lru_cache` 或自管缓存）。

# 十三、稳健性与安全域

* 统一 **safe** 操作：`safe_div`, `safe_log`, `safe_pow`；
* 全局 `np.errstate` 与结果 `np.nan_to_num`、`np.isfinite` 过滤；
* 训练前后 **值域扫描报告**（最大/最小/非有限比率）并入日志。

# 十四、（可选）物理量纲一致性

* 为每个原子特征定义量纲向量，操作符携带量纲变换规则；
* 生成时即过滤量纲非法表达式；或在 SO 目标中加入量纲惩罚。

# 十五、配置与可重复性

* 统一 `Config`：可由 `dict`/YAML 初始化；
* `seed_all(random_state)` 固定 NumPy/Sklearn 随机性；
* 版本与依赖写入报告，保证复现实验。

# 十六、测试与验证

* **单元测试**：

  * 操作符边界（除零/负数开方/对数域）；
  * 生成器去重与复杂度计数；
  * SIS/ SO 与小样例对拍（合成线性/非线性真值）。
* **基准测试**：不同 K、不同操作符集下的时间/内存曲线。
* **一致性测试**：多次运行在同一随机种子下结果恒定。

# 十七、依赖清单（仅 Python 科学计算库）

* **必需**：`numpy`, `scipy`, `scikit-learn`, `pandas`
* **可选**：`numba`（加速）、`sympy`（公式导出/化简）、`joblib`（并行）

# 十八、里程碑

1. M1：DSL/操作符与安全求值；K=2 的基础生成+SIS（皮尔森）
2. M2：SO（OMP/Lasso），报告与日志；最佳公式导出
3. M3：K≥3 的可扩展生成、并行与缓存、互信息/HSIC
4. M4：量纲检查、稳定性与大规模性能优化
5. M5：文档、单元/基准测试、CLI

