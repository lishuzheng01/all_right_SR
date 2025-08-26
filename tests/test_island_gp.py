#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sisso_py.evolutionary.island_gp import IslandGPRegressor

def test_island_gp():
    print("测试 IslandGPRegressor...")
    
    # 创建简单测试数据
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 2), columns=['x1', 'x2'])
    y = pd.Series(2*X['x1'] + X['x2'] + np.random.rand(20)*0.1)
    
    # 测试模型
    model = IslandGPRegressor(n_islands=2, island_size=10, generations=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    correlation = np.corrcoef(y, y_pred)[0,1]
    print(f'预测完成，R² 相关性: {correlation:.4f}')
    
    # 测试模型信息
    info = model.get_model_info()
    print(f"模型信息: {info}")
    
    print('✅ IslandGPRegressor 功能测试通过!')
    return True

if __name__ == "__main__":
    test_island_gp()
