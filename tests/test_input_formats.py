# -*- coding: utf-8 -*-
"""
测试多种输入格式支持
"""

import numpy as np
import pandas as pd
from sisso_py.sparse_regression.sisso import SISSORegressor
from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor

def test_numpy_input():
    """测试numpy数组输入"""
    print("=== 测试numpy数组输入 ===")
    
    # 生成测试数据
    X_array = np.linspace(-2*np.pi, 2*np.pi, 100)
    y_array = np.sin(X_array) + 0.1 * np.random.randn(100)
    
    print(f"X 类型: {type(X_array)}, 形状: {X_array.shape}")
    print(f"y 类型: {type(y_array)}, 形状: {y_array.shape}")
    
    # 测试 SISSO
    print("\n--- 测试 SISSO ---")
    try:
        model = SISSORegressor(K=2, sis_screener='pearson', so_solver='omp')
        model.fit(X_array, y_array)
        
        # 预测
        y_pred = model.predict(X_array)
        print(f"预测结果形状: {y_pred.shape}")
        print("SISSO numpy输入测试成功!")
        
    except Exception as e:
        print(f"SISSO numpy输入测试失败: {e}")
    
    # 测试 Lasso 
    print("\n--- 测试 Lasso ---")
    try:
        lasso_model = LassoRegressor(poly_degree=2)
        lasso_model.fit(X_array, y_array)
        
        # 预测
        y_pred_lasso = lasso_model.predict(X_array)
        print(f"Lasso预测结果形状: {y_pred_lasso.shape}")
        print("Lasso numpy输入测试成功!")
        
    except Exception as e:
        print(f"Lasso numpy输入测试失败: {e}")

def test_pandas_input():
    """测试pandas对象输入"""
    print("\n=== 测试pandas对象输入 ===")
    
    # 生成测试数据
    X_array = np.linspace(-2*np.pi, 2*np.pi, 100)
    y_array = np.sin(X_array) + 0.1 * np.random.randn(100)
    
    X_df = pd.DataFrame(X_array.reshape(-1, 1), columns=['x'])
    y_series = pd.Series(y_array, name='y')
    
    print(f"X 类型: {type(X_df)}, 形状: {X_df.shape}")
    print(f"y 类型: {type(y_series)}, 形状: {y_series.shape}")
    
    # 测试 SISSO
    print("\n--- 测试 SISSO ---")
    try:
        model = SISSORegressor(K=2, sis_screener='pearson', so_solver='omp')
        model.fit(X_df, y_series)
        
        # 预测
        y_pred = model.predict(X_df)
        print(f"预测结果形状: {y_pred.shape}")
        print("SISSO pandas输入测试成功!")
        
    except Exception as e:
        print(f"SISSO pandas输入测试失败: {e}")
    
    # 测试 Lasso
    print("\n--- 测试 Lasso ---")
    try:
        lasso_model = LassoRegressor(poly_degree=2)
        lasso_model.fit(X_df, y_series)
        
        # 预测
        y_pred_lasso = lasso_model.predict(X_df)
        print(f"Lasso预测结果形状: {y_pred_lasso.shape}")
        print("Lasso pandas输入测试成功!")
        
    except Exception as e:
        print(f"Lasso pandas输入测试失败: {e}")

def test_mixed_input():
    """测试混合输入（训练用numpy，预测用pandas）"""
    print("\n=== 测试混合输入 ===")
    
    # 训练数据 (numpy)
    X_train = np.linspace(-2*np.pi, 2*np.pi, 100)
    y_train = np.sin(X_train) + 0.1 * np.random.randn(100)
    
    # 测试数据 (pandas)
    X_test_array = np.linspace(-np.pi, np.pi, 50)
    X_test_df = pd.DataFrame(X_test_array.reshape(-1, 1), columns=['x'])
    
    print("训练: numpy数组，预测: pandas DataFrame")
    
    try:
        # 训练
        model = SISSORegressor(K=2, sis_screener='pearson', so_solver='omp')
        model.fit(X_train, y_train, feature_names=['x'])
        
        # 预测
        y_pred = model.predict(X_test_df)
        print(f"混合输入预测结果形状: {y_pred.shape}")
        print("混合输入测试成功!")
        
    except Exception as e:
        print(f"混合输入测试失败: {e}")

if __name__ == "__main__":
    test_numpy_input()
    test_pandas_input()
    test_mixed_input()
    print("\n所有测试完成!")
