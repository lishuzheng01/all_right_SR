import numpy as np
import pandas as pd
from sisso_py import SissoRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子确保可重复性
np.random.seed(42)

# 准备更大的数据集
n_samples = 500  # 增加样本数量

X = pd.DataFrame({
    'x1': np.random.randn(n_samples),
    'x2': np.random.randn(n_samples),
    'x3': np.random.randn(n_samples)
})

# 简化目标函数，减少噪声
y = 2 * X['x1']**2 + 3 * X['x2'] - X['x3'] + np.random.randn(n_samples) * 0.05

print("=== 数据信息 ===")
print(f"样本数量: {n_samples}")
print(f"真实公式: y = 2*x1² + 3*x2 - x3 + noise")
print(f"噪声水平: 0.05")

# 创建更保守的模型配置
model = SissoRegressor(
    K=2,  # 降低复杂度
    operators=['+', '-', '*', 'safe_div', 'square'],  # 移除log和sqrt以避免数值问题
    sis_screener='pearson',
    sis_topk=500,  # 减少特征数量
    so_solver='omp',
    so_max_terms=3,  # 减少最终项数
    cv=5,
    random_state=42
)

print("\n=== 开始训练 ===")
model.fit(X, y)

# 预测和评估
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"\n=== 模型性能 ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 获取详细报告
report = model.explain()

print(f"\n=== 发现的公式 ===")
print("LaTeX格式:")
print(report['results']['final_model']['formula_latex'])

print(f"\n=== 选中的特征 ===")
features = report['results']['final_model']['features']
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature['signature']}")
    print(f"   系数: {feature['coefficient']:.4f}")
    print(f"   复杂度: {feature['complexity']}")

print(f"\n=== 运行统计 ===")
run_info = report['run_info']
print(f"生成特征总数: {run_info['total_features_generated']}")
print(f"SIS后特征数: {run_info['features_after_sis']}")
print(f"最终模型特征数: {run_info['features_in_final_model']}")

print(f"\n=== 理论对比 ===")
print("期望发现的模式:")
print("- x1的平方项 (系数≈2)")
print("- x2的线性项 (系数≈3)")  
print("- x3的线性项 (系数≈-1)")
