
import numpy as np
import pandas as pd
import logging
import traceback

# 配置日志记录，以便在测试期间看到详细信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from sisso_py.sparse_regression.sisso import SISSORegressor
    from sisso_py.dsl.dimension import Dimension
    print("模块导入成功。")
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保您已经正确安装了sisso_py库，或者项目路径已添加到PYTHONPATH。")
    exit()

def generate_test_data(n_samples=50, n_features=3):
    """生成小规模的测试数据，避免数据量过大影响测试速度"""
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(n_samples, n_features) * 5, 
                     columns=[f'x{i+1}' for i in range(n_features)])
    # 定义一个简单的非线性关系，以便测试可以发现某些东西
    y = pd.Series(1.5 * X['x1']**2 + 0.8 * np.log(X['x2'] + 1) + np.sin(X['x3']) + 
                  np.random.randn(n_samples) * 0.05)
    
    # 定义量纲信息
    # BASE_DIMENSIONS in sisso_py is ['M', 'L', 'T', 'I', 'Θ', 'N', 'J']
    feature_dims = {
        'x1': Dimension([0, 1, 0, 0, 0, 0, 0]),      # L
        'x2': Dimension([0, 0, 2, 0, 0, 0, 0]),      # T^2
        'x3': Dimension([1, 0, 0, 0, -1, 0, 0])   # M/Θ
    }
    # 假设 y 的量纲是 x1^2 的量纲 (L^2)
    target_dim = Dimension([0, 2, 0, 0, 0, 0, 0])
    
    return X, y, feature_dims, target_dim

def run_test(test_name: str, model: SISSORegressor, X: pd.DataFrame, y: pd.Series, 
             feature_dims: dict = None, target_dim: Dimension = None):
    """
    运行单个测试用例并捕获异常。
    """
    print(f"\n{'='*20} 开始测试: {test_name} {'='*20}")
    try:
        # 拟合模型
        if model.dimensional_check:
            print("  -> 启用量纲检查进行拟合...")
            if feature_dims is None or target_dim is None:
                raise ValueError("量纲检查需要 feature_dims 和 target_dim。")
            model.fit(X, y, feature_dimensions=feature_dims, target_dimension=target_dim)
        else:
            print("  -> 禁言量纲检查进行拟合...")
            model.fit(X, y)
        print("  [成功] 模型拟合完成。")

        # 预测
        y_pred = model.predict(X)
        print(f"  [成功] 模型预测完成。预测样本: {y_pred[:3]}")
        assert isinstance(y_pred, np.ndarray), "预测结果应为 numpy.ndarray"

        # 获取模型信息
        model_info = model.get_model_info()
        print("  [成功] 获取模型信息完成。")
        print(f"  模型公式: {model_info.get('formula', '未能生成公式')}")
        assert 'formula' in model_info, "模型信息应包含 'formula'"

        print(f"--- 测试 '{test_name}' 通过 ---")
        return True

    except Exception as e:
        print(f"  [失败] 测试 '{test_name}' 发生错误: {e}")
        print(traceback.format_exc())
        return False
    finally:
        print(f"{'='*20} 测试结束: {test_name} {'='*20}\n")


def main():
    """
    主测试函数，调用所有API接口。
    """
    print("正在生成测试数据...")
    X, y, feature_dims, target_dim = generate_test_data()
    print(f"数据生成完毕，X shape: {X.shape}, y shape: {y.shape}")

    all_tests_passed = True
    
    # 测试配置
    # 注意：为了快速检测，K和sis_topk设置得较小
    base_params = {
        'K': 2,
        'sis_topk': 50,
        'so_max_terms': 2,
        'cv': 3,
        'n_jobs': 1, # 在某些系统上，-1可能会导致问题，因此设为1以确保稳定性
        'random_state': 42
    }

    # 1. 默认配置测试 (pearson + omp)
    model_default = SISSORegressor(**base_params)
    if not run_test("默认配置 (pearson + omp)", model_default, X, y):
        all_tests_passed = False

    # 2. 量纲检查测试
    model_dim = SISSORegressor(**base_params, dimensional_check=True)
    if not run_test("量纲检查", model_dim, X, y, feature_dims, target_dim):
        all_tests_passed = False

    # 3. 遍历不同的求解器和筛选器
    solvers = ['omp', 'lasso', 'elasticnet']
    screeners = ['pearson', 'f_regression'] # 'mutual_info' 较慢，暂不加入快速测试

    for solver in solvers:
        for screener in screeners:
            test_name = f"求解器: {solver}, 筛选器: {screener}"
            params = {**base_params, 'so_solver': solver, 'sis_screener': screener}
            model = SISSORegressor(**params)
            if not run_test(test_name, model, X, y):
                all_tests_passed = False

    # 总结
    print("\n" + "="*50)
    if all_tests_passed:
        print("🎉 全部API接口测试通过！功能看起来是完善的。")
    else:
        print("🔥 部分API接口测试失败！请检查上面的终端输出以定位问题。")
    print("="*50)


if __name__ == "__main__":
    main()
