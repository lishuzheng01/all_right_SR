from sisso_py.sparse_regression.sisso import SISSORegressor
import numpy as np

# 使用numpy数组直接输入
X = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(X)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X type:", type(X))
print("y type:", type(y))

model = SISSORegressor(K=3, sis_screener='pearson', so_solver='omp')
model.fit(X, y)

report = model.explain()
print(report)