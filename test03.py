import numpy as np
import pandas as pd
from sisso_py import SissoRegressor

# 准备数据
X = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.sin(np.random.randn(100)),
    'x3': np.cos(np.random.randn(100))
})


y = 2 * X['x1']**2 + 3 * X['x2'] - X['x3'] + np.random.randn(100) * 0.1

# 创建和训练模型
model = SissoRegressor(
    K=1,  # 最大复杂度层数
    operators=['+', '-', '*', 'safe_div', 'sqrt', 'square', 'log','exp','abs','sin','cos','tan','sinh','cosh','tanh','reciprocal'],
    sis_screener='random',  # 筛选方法
    sis_topk=1000,  # SIS 保留特征数
    so_solver='omp',  # 稀疏求解器
    so_max_terms=3,  # 最终模型最大项数
    cv=5  # 交叉验证折数
)

# 拟合模型
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 获取解释和报告
report = model.explain()
print(report)