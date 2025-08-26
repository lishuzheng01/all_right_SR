#!/usr/bin/env python3
"""
LASSO独立测试脚本
"""

import time
import logging
import numpy as np
import pandas as pd
from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sisso_py.dsl.dimension import Dimension

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """生成与测试框架相同的测试数据"""
    np.random.seed(42)
    n_samples, n_features = 100, 4
    
    # 生成基础特征数据
    X_np = np.random.rand(n_samples, n_features) * 10 - 5  # [-5, 5] 范围
    X = pd.DataFrame(X_np, columns=[f'x{i+1}' for i in range(n_features)])
    
    # 确保正值用于对数和开方运算
    X_positive_np = np.abs(X_np) + 0.1
    
    # 中等复杂度：多项式 + 三角函数
    y = (1.5 * X_np[:, 0]**2 + 0.8 * np.log(X_positive_np[:, 1]) + 
         np.sin(X_np[:, 2]) + 0.3 * X_np[:, 3] + np.random.randn(n_samples) * 0.1)
    
    y = pd.Series(y, name='target')
    
    logger.info(f"数据生成完成: X={X.shape}, y={y.shape}")
    logger.info(f"X列: {list(X.columns)}")
    logger.info(f"y统计: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
    
    return X, y

def test_lasso_standalone():
    """独立测试LASSO模型"""
    logger.info("=" * 80)
    logger.info("开始LASSO独立测试")
    logger.info("=" * 80)
    
    # 生成测试数据
    X, y = generate_test_data()
    
    # 使用与测试框架完全相同的参数
    sparse_params = {
        'alpha': 0.01,
        'max_iter': 1000,
        'poly_degree': 2,
        'normalize': True
    }
    
    logger.info(f"测试参数: {sparse_params}")
    
    start_time = time.time()
    
    try:
        logger.info("🧪 开始测试: LASSO稀疏回归")
        
        # 创建模型实例
        model = LassoRegressor(**sparse_params)
        
        # 拟合模型
        logger.info("开始模型训练...")
        model.fit(X, y)
        logger.info("模型训练完成")
        
        # 预测
        logger.info("开始预测...")
        y_pred = model.predict(X)
        logger.info(f"预测完成，结果形状: {y_pred.shape}")
        
        # 计算性能指标
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ 测试成功!")
        logger.info(f"   - 执行时间: {execution_time:.3f}s")
        logger.info(f"   - MSE: {mse:.6f}")
        logger.info(f"   - R²: {r2:.6f}")
        
        # 获取模型信息
        model_info = model.get_model_info()
        logger.info(f"   - 非零系数: {model_info.get('nonzero_terms', 'N/A')}")
        logger.info(f"   - 公式: {model_info.get('formula', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ 测试失败!")
        logger.error(f"   - 执行时间: {execution_time:.3f}s")
        logger.error(f"   - 错误信息: {str(e)}")
        
        import traceback
        logger.error("完整错误堆栈:")
        logger.error(traceback.format_exc())
        
        return False

def test_lasso_edge_cases():
    """测试LASSO的边界情况"""
    logger.info("=" * 80)
    logger.info("测试LASSO边界情况")
    logger.info("=" * 80)
    
    # 生成各种测试数据
    test_cases = [
        ("正常数据", lambda: generate_test_data()),
        ("小数据集", lambda: (
            pd.DataFrame(np.random.rand(10, 4), columns=['x1', 'x2', 'x3', 'x4']),
            pd.Series(np.random.rand(10), name='target')
        )),
        ("最小数据集", lambda: (
            pd.DataFrame(np.random.rand(3, 4), columns=['x1', 'x2', 'x3', 'x4']),
            pd.Series(np.random.rand(3), name='target')
        )),
    ]
    
    for case_name, data_generator in test_cases:
        logger.info(f"测试情况: {case_name}")
        
        try:
            X, y = data_generator()
            
            if X.empty or len(X) == 0:
                logger.warning(f"  跳过空数据集: {case_name}")
                continue
                
            logger.info(f"  数据形状: X={X.shape}, y={y.shape}")
            
            model = LassoRegressor(alpha=0.01, poly_degree=2, normalize=True)
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            logger.info(f"  ✅ {case_name} 成功: R²={r2:.4f}")
            
        except Exception as e:
            logger.error(f"  ❌ {case_name} 失败: {str(e)}")

if __name__ == "__main__":
    # 运行测试
    success = test_lasso_standalone()
    
    if success:
        test_lasso_edge_cases()
        logger.info("所有测试完成!")
    else:
        logger.error("主要测试失败，跳过边界测试")
