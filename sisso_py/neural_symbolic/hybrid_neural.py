# -*- coding: utf-8 -*-
"""
神经符号混合方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class NeuralSymbolicHybrid(BaseEstimator, RegressorMixin):
    """
    神经网络与符号回归的混合方法
    
    结合神经网络的函数逼近能力和符号回归的可解释性，
    使用注意力机制融合两个组件的输出。
    """
    
    def __init__(self,
                 neural_component='transformer',
                 symbolic_component='gp',
                 fusion_method='attention',
                 epochs=30,
                 hidden_dim=64,
                 n_heads=4,
                 n_layers=2,
                 dropout=0.1,
                 alpha=0.5,  # 神经网络和符号回归的权重
                 max_symbolic_depth=5,
                 symbolic_population=50,
                 symbolic_generations=20,
                 random_state=42):
        
        self.neural_component = neural_component
        self.symbolic_component = symbolic_component
        self.fusion_method = fusion_method
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.alpha = alpha
        self.max_symbolic_depth = max_symbolic_depth
        self.symbolic_population = symbolic_population
        self.symbolic_generations = symbolic_generations
        self.random_state = random_state
        
        self.neural_model_ = None
        self.symbolic_model_ = None
        self.fusion_weights_ = None
        self.best_expression_ = None
        self.training_history_ = []
        
    def fit(self, X, y, feature_names=None):
        """
        训练神经符号混合模型
        """
        logger.info("开始神经符号混合模型训练...")
        
        np.random.seed(self.random_state)
        
        self.feature_names_ = feature_names or [f'x{i}' for i in range(X.shape[1])]
        self.n_features_ = X.shape[1]
        
        # 第一阶段：训练神经网络组件
        logger.info("训练神经网络组件...")
        self._train_neural_component(X, y)
        
        # 第二阶段：训练符号回归组件
        logger.info("训练符号回归组件...")
        self._train_symbolic_component(X, y)
        
        # 第三阶段：学习融合权重
        logger.info("学习融合权重...")
        self._learn_fusion_weights(X, y)
        
        logger.info("神经符号混合模型训练完成")
        return self
    
    def predict(self, X):
        """预测"""
        if self.neural_model_ is None or self.symbolic_model_ is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 获取神经网络预测
        neural_pred = self._neural_predict(X)
        
        # 获取符号回归预测
        symbolic_pred = self._symbolic_predict(X)
        
        # 融合预测结果
        if self.fusion_method == 'weighted':
            final_pred = self.alpha * neural_pred + (1 - self.alpha) * symbolic_pred
        elif self.fusion_method == 'attention':
            final_pred = self._attention_fusion(X, neural_pred, symbolic_pred)
        else:
            # 默认使用加权平均
            final_pred = 0.5 * neural_pred + 0.5 * symbolic_pred
        
        return final_pred
    
    def _train_neural_component(self, X, y):
        """训练神经网络组件"""
        # 确保X是numpy数组
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # 简化的神经网络训练（实际实现需要深度学习框架）
        self.neural_model_ = {
            'type': self.neural_component,
            'weights': [np.random.randn(self.n_features_, self.hidden_dim),
                       np.random.randn(self.hidden_dim, 1)],
            'biases': [np.random.randn(self.hidden_dim), np.random.randn(1)]
        }
        
        # 模拟训练过程
        for epoch in range(self.epochs):
            # 前向传播
            neural_pred = self._neural_predict(X)
            loss = mean_squared_error(y, neural_pred)
            
            # 简单的权重更新（实际应使用反向传播）
            noise_scale = 0.01 * (1 - epoch / self.epochs)  # 逐渐减小噪声
            for i in range(len(self.neural_model_['weights'])):
                self.neural_model_['weights'][i] += np.random.randn(*self.neural_model_['weights'][i].shape) * noise_scale
                self.neural_model_['biases'][i] += np.random.randn(*self.neural_model_['biases'][i].shape) * noise_scale
            
            if epoch % 5 == 0:
                logger.debug(f"Neural epoch {epoch}, loss: {loss:.4f}")
    
    def _train_symbolic_component(self, X, y):
        """训练符号回归组件"""
        # 简化的符号回归训练
        best_expression = None
        best_score = float('inf')
        
        for generation in range(self.symbolic_generations):
            # 生成随机表达式种群
            population = self._generate_symbolic_population()
            
            # 评估每个表达式
            for expression in population:
                try:
                    pred = self._evaluate_symbolic_expression(expression, X)
                    score = mean_squared_error(y, pred)
                    
                    if score < best_score:
                        best_score = score
                        best_expression = expression
                        
                except:
                    continue
            
            if generation % 5 == 0:
                logger.debug(f"Symbolic generation {generation}, best score: {best_score:.4f}")
        
        self.symbolic_model_ = {
            'expression': best_expression,
            'score': best_score
        }
        self.best_expression_ = best_expression
    
    def _learn_fusion_weights(self, X, y):
        """学习融合权重"""
        neural_pred = self._neural_predict(X)
        symbolic_pred = self._symbolic_predict(X)
        
        if self.fusion_method == 'attention':
            # 简化的注意力权重学习
            feature_importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])
            attention_weights = feature_importance / np.sum(feature_importance)
            self.fusion_weights_ = attention_weights
        else:
            # 基于验证误差选择权重
            neural_mse = mean_squared_error(y, neural_pred)
            symbolic_mse = mean_squared_error(y, symbolic_pred)
            
            # 权重与误差成反比
            neural_weight = 1 / (neural_mse + 1e-8)
            symbolic_weight = 1 / (symbolic_mse + 1e-8)
            total_weight = neural_weight + symbolic_weight
            
            self.alpha = neural_weight / total_weight
    
    def _neural_predict(self, X):
        """神经网络预测"""
        if self.neural_model_ is None:
            return np.zeros(X.shape[0])
        
        # 确保X是numpy数组
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # 简单的前向传播
        h = np.tanh(X_array @ self.neural_model_['weights'][0] + self.neural_model_['biases'][0])
        output = h @ self.neural_model_['weights'][1] + self.neural_model_['biases'][1]
        return output.flatten()
    
    def _symbolic_predict(self, X):
        """符号回归预测"""
        if self.symbolic_model_ is None or self.symbolic_model_['expression'] is None:
            return np.zeros(X.shape[0])
        
        return self._evaluate_symbolic_expression(self.symbolic_model_['expression'], X)
    
    def _attention_fusion(self, X, neural_pred, symbolic_pred):
        """注意力机制融合"""
        if self.fusion_weights_ is None:
            return 0.5 * neural_pred + 0.5 * symbolic_pred
        
        # 基于特征重要性的注意力融合
        feature_context = np.mean(X, axis=0)
        context_score = np.dot(feature_context, self.fusion_weights_)
        attention_alpha = 1 / (1 + np.exp(-context_score))  # sigmoid
        
        return attention_alpha * neural_pred + (1 - attention_alpha) * symbolic_pred
    
    def _generate_symbolic_population(self):
        """生成符号表达式种群"""
        population = []
        
        for _ in range(self.symbolic_population):
            expression = self._generate_random_expression()
            population.append(expression)
        
        return population
    
    def _generate_random_expression(self):
        """生成随机符号表达式"""
        templates = [
            "({var1} + {var2})",
            "({var1} * {var2})",
            "({var1} - {var2})",
            "sin({var1})",
            "cos({var1})",
            "exp({var1} / 5)",
            "log(abs({var1}) + 1)",
            "({var1} ** 2)",
            "sqrt(abs({var1}))",
            "({var1} + {var2}) * {var3}",
            "sin({var1}) + cos({var2})"
        ]
        
        template = np.random.choice(templates)
        
        # 随机选择变量
        variables = {f'var{i+1}': np.random.choice(self.feature_names_) 
                    for i in range(3)}
        
        return template.format(**variables)
    
    def _evaluate_symbolic_expression(self, expression, X):
        """评估符号表达式"""
        try:
            # 创建局部环境
            local_env = {}
            for i, name in enumerate(self.feature_names_):
                local_env[name] = X[:, i]
            
            # 添加数学函数
            local_env.update({
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-8),
                'sqrt': lambda x: np.sqrt(np.abs(x)),
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e
            })
            
            result = eval(expression, {"__builtins__": {}}, local_env)
            
            if np.isscalar(result):
                result = np.full(X.shape[0], result)
            
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
        except:
            return np.zeros(X.shape[0])
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'formula': self.best_expression_,
            'method': 'Neural-Symbolic Hybrid',
            'neural_component': self.neural_component,
            'symbolic_component': self.symbolic_component,
            'fusion_method': self.fusion_method,
            'fusion_alpha': self.alpha,
            'symbolic_score': self.symbolic_model_['score'] if self.symbolic_model_ else None
        }
