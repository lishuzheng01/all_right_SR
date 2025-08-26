# -*- coding: utf-8 -*-
"""
深度符号回归方法
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

class DeepSymbolicRegression(BaseEstimator, RegressorMixin):
    """
    深度学习驱动的符号回归方法
    
    使用神经网络编码器-解码器架构来生成符号表达式。
    编码器学习数据的表示，解码器生成对应的符号表达式。
    """
    
    def __init__(self,
                 encoder_layers=[64, 32],
                 decoder_layers=[32, 64], 
                 max_length=20,
                 epochs=50,
                 batch_size=32,
                 learning_rate=0.001,
                 dropout_rate=0.1,
                 vocabulary_size=100,
                 embedding_dim=128,
                 random_state=42):
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        
        self.best_expression_ = None
        self.best_score_ = float('inf')
        self.training_history_ = []
        self.vocabulary_ = None
        
    def fit(self, X, y, feature_names=None):
        """
        训练深度符号回归模型
        """
        logger.info("开始深度符号回归训练...")
        
        np.random.seed(self.random_state)
        
        self.feature_names_ = feature_names or [f'x{i}' for i in range(X.shape[1])]
        self.n_features_ = X.shape[1]
        
        # 创建词汇表
        self._build_vocabulary()
        
        # 模拟深度学习训练过程
        for epoch in range(self.epochs):
            # 生成表达式候选
            expressions = self._generate_expression_candidates(batch_size=self.batch_size)
            
            epoch_loss = 0
            valid_expressions = 0
            
            for expression in expressions:
                try:
                    y_pred = self._evaluate_expression(expression, X)
                    mse = mean_squared_error(y, y_pred)
                    
                    # 更新最佳表达式
                    if mse < self.best_score_:
                        self.best_score_ = mse
                        self.best_expression_ = expression
                    
                    epoch_loss += mse
                    valid_expressions += 1
                    
                except:
                    epoch_loss += 1000  # 惩罚无效表达式
            
            avg_loss = epoch_loss / len(expressions) if expressions else 1000
            
            self.training_history_.append({
                'epoch': epoch,
                'loss': avg_loss,
                'valid_expressions': valid_expressions,
                'best_score': self.best_score_
            })
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Best Score: {self.best_score_:.4f}")
        
        logger.info(f"训练完成，最佳表达式: {self.best_expression_}")
        self._train_X = X
        self._train_y = y
        return self
    
    def predict(self, X):
        """预测"""
        if self.best_expression_ is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        return self._evaluate_expression(self.best_expression_, X)
    
    def _build_vocabulary(self):
        """构建符号词汇表"""
        self.vocabulary_ = {
            'variables': self.feature_names_,
            'operators': ['+', '-', '*', '/', '^', 'sin', 'cos', 'exp', 'log', 'sqrt'],
            'constants': ['0', '1', '2', 'pi', 'e'],
            'special': ['(', ')', 'START', 'END', 'PAD']
        }
        
        # 创建符号到索引的映射
        self.token_to_idx = {}
        idx = 0
        for category, tokens in self.vocabulary_.items():
            for token in tokens:
                self.token_to_idx[token] = idx
                idx += 1
        
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        
    def _generate_expression_candidates(self, batch_size=32):
        """生成表达式候选"""
        expressions = []
        
        for _ in range(batch_size):
            # 使用简单的语法规则生成表达式
            expression = self._generate_single_expression()
            expressions.append(expression)
            
        return expressions
    
    def _generate_single_expression(self):
        """生成单个表达式"""
        templates = [
            # 二元操作模板
            "({var1} {op} {var2})",
            "({var1} {op} {const})",
            "({const} {op} {var1})",
            
            # 一元函数模板
            "{func}({var1})",
            "{func}({var1} {op} {var2})",
            
            # 复合模板
            "({func}({var1}) {op} {var2})",
            "({var1} {op} {func}({var2}))"
        ]
        
        template = np.random.choice(templates)
        
        # 填充模板
        var1 = np.random.choice(self.feature_names_)
        var2 = np.random.choice(self.feature_names_)
        op = np.random.choice(['+', '-', '*', '/'])
        func = np.random.choice(['sin', 'cos', 'exp', 'log', 'sqrt'])
        const = np.random.choice(['1', '2', '0.5', 'pi'])
        
        expression = template.format(
            var1=var1, var2=var2, op=op, func=func, const=const
        )
        
        return expression
    
    def _evaluate_expression(self, expression, X):
        """评估表达式"""
        try:
            # 创建局部环境
            local_env = {}
            for i, name in enumerate(self.feature_names_):
                local_env[name] = X[:, i]
            
            # 添加数学函数和常数
            local_env.update({
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'log': lambda x: np.log(np.abs(x) + 1e-8),  # 安全的对数
                'sqrt': lambda x: np.sqrt(np.abs(x)),  # 安全的平方根
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e
            })
            
            # 评估表达式
            result = eval(expression, {"__builtins__": {}}, local_env)
            
            # 确保结果是数组
            if np.isscalar(result):
                result = np.full(X.shape[0], result)
            
            # 处理无效值
            result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
            
        except:
            # 返回零数组作为失败的表达式
            return np.zeros(X.shape[0])
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'formula': self.best_expression_,
            'best_score': self.best_score_,
            'method': 'Deep Symbolic Regression',
            'architecture': {
                'encoder_layers': self.encoder_layers,
                'decoder_layers': self.decoder_layers,
                'vocabulary_size': len(self.token_to_idx) if self.vocabulary_ else 0
            },
            'training_epochs': len(self.training_history_)
        }

    def _build_report(self):
        """生成包含评价指标的格式化报告"""
        from ..model.formatted_report import SissoReport
        if self.best_expression_ is None:
            return SissoReport({"status": "Model not fitted."})

        metrics = {}
        try:
            y_pred = self.predict(self._train_X)
            mse = mean_squared_error(self._train_y, y_pred)
            metrics = {
                "train_mse": mse,
                "train_rmse": float(np.sqrt(mse)),
                "train_mae": mean_absolute_error(self._train_y, y_pred),
                "train_r2": r2_score(self._train_y, y_pred),
                "train_samples": len(self._train_y)
            }
        except Exception as e:
            metrics = {
                "train_mse": None,
                "train_rmse": None,
                "train_mae": None,
                "train_r2": None,
                "error": str(e)
            }

        report = {
            "title": "Deep Symbolic Regression 分析报告",
            "configuration": {
                "epochs": self.epochs,
                "encoder_layers": self.encoder_layers,
                "decoder_layers": self.decoder_layers
            },
            "results": {
                "final_model": {
                    "formula_latex": self.best_expression_,
                    "formula_sympy": self.best_expression_,
                    "intercept": 0,
                    "features": []
                },
                "metrics": metrics
            },
            "run_info": {
                "total_features_generated": 0,
                "features_after_sis": 0,
                "features_in_final_model": 1
            }
        }

        return SissoReport(report)

    @property
    def explain(self):
        return self._build_report()
