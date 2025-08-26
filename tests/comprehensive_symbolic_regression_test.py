# -*- coding: utf-8 -*-
"""
全面的符号回归方法测试脚本
包含五大类符号回归方法的全面测试

一、进化算法类
二、稀疏建模与筛选类
三、贝叶斯与概率模型类
四、强化学习与神经符号方法
五、混合与新兴方法
"""

import pytest

# 本模块为人工测试脚本，在自动化测试环境下直接跳过
pytest.skip("comprehensive manual suite", allow_module_level=True)

import numpy as np
import pandas as pd
import logging
import traceback
import warnings
from typing import Dict, Any, List, Tuple
import time

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略一些常见的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from sisso_py.sparse_regression.sisso import SISSORegressor
    from sisso_py.evolutionary.gp import GeneticProgramming
    from sisso_py.evolutionary.ga_pso import GAPSORegressor
    from sisso_py.evolutionary.island_gp import IslandGPRegressor 
    from sisso_py.probabilistic.bsr import BayesianSymbolicRegressor
    from sisso_py.probabilistic.ppi import ProbabilisticProgramInduction
    from sisso_py.dsl.dimension import Dimension
    from sisso_py.sparse_regression.lasso_ridge_omp import LassoRegressor
    from sisso_py.sparse_regression.sindy import SINDyRegressor
    from sisso_py.utils.logging import setup_logging
    print("所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保您已经正确安装了sisso_py库")
    exit()


class SymbolicRegressionTestSuite:
    """符号回归方法测试套件"""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = None
        self.feature_dims = None
        self.target_dim = None
        
    def generate_test_data(self, n_samples=100, n_features=4, complexity='medium'):
        """生成不同复杂度的测试数据"""
        np.random.seed(42)
        
        # 生成基础特征数据
        X_np = np.random.rand(n_samples, n_features) * 10 - 5  # [-5, 5] 范围
        X = pd.DataFrame(X_np, columns=[f'x{i+1}' for i in range(n_features)])
        
        # 确保正值用于对数和开方运算
        X_positive_np = np.abs(X_np) + 0.1
        
        if complexity == 'simple':
            # 简单线性关系
            y = 2.5 * X_np[:, 0] + 1.8 * X_np[:, 1] - 0.5 * X_np[:, 2] + np.random.randn(n_samples) * 0.1
        elif complexity == 'medium':
            # 中等复杂度：多项式 + 三角函数
            y = (1.5 * X_np[:, 0]**2 + 0.8 * np.log(X_positive_np[:, 1]) + 
                 np.sin(X_np[:, 2]) + 0.3 * X_np[:, 3] + np.random.randn(n_samples) * 0.1)
        elif complexity == 'high':
            # 高复杂度：包含多种函数形式
            y = (2.0 * X_np[:, 0]**3 + np.exp(X_np[:, 1]/5) + 
                 np.sqrt(X_positive_np[:, 2]) * np.cos(X_np[:, 3]) + 
                 1.2 * X_np[:, 0] * X_np[:, 1] + np.random.randn(n_samples) * 0.2)
        else:
            raise ValueError("复杂度必须是 'simple', 'medium', 或 'high'")
        
        y = pd.Series(y, name='target')
        
        # 定义量纲信息
        feature_dims = {
            'x1': Dimension([0, 1, 0, 0, 0, 0, 0]),    # L (长度)
            'x2': Dimension([0, 0, 1, 0, 0, 0, 0]),    # T (时间)
            'x3': Dimension([1, 0, 0, 0, 0, 0, 0]),    # M (质量)
            'x4': Dimension([0, 0, 0, 1, 0, 0, 0])     # I (电流)
        }
        target_dim = Dimension([0, 2, 0, 0, 0, 0, 0])  # L^2
        
        self.test_data = (X, y)
        self.feature_dims = feature_dims
        self.target_dim = target_dim
        
        return X, y, feature_dims, target_dim
    
    def run_single_test(self, test_name: str, model_class, model_params: Dict[str, Any], 
                       use_dimensions: bool = False) -> Dict[str, Any]:
        """运行单个测试用例"""
        logger.info(f"🧪 开始测试: {test_name}")
        start_time = time.time()
        
        try:
            # 创建模型实例
            model = model_class(**model_params)
            
            # 获取测试数据
            if self.test_data is None:
                raise ValueError("测试数据未初始化")
            X, y = self.test_data
            
            # 拟合模型
            if use_dimensions and hasattr(model, 'fit') and 'feature_dimensions' in model.fit.__code__.co_varnames:
                model.fit(X, y, feature_dimensions=self.feature_dims, target_dimension=self.target_dim)
            else:
                if hasattr(model, 'fit'):
                    # 检查模型是否真的需要feature_names参数
                    import inspect
                    try:
                        sig = inspect.signature(model.fit)
                        if 'feature_names' in sig.parameters:
                            model.fit(X, y, feature_names=X.columns.tolist())
                        else:
                            model.fit(X, y)
                    except:
                        # 如果签名检查失败，使用基本的fit方法
                        model.fit(X, y)
                else:
                    raise AttributeError(f"模型 {model_class.__name__} 没有 fit 方法")
            
            # 预测
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                
                # 计算性能指标
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
            else:
                y_pred = None
                mse = None
                r2 = None
            
            # 获取模型信息
            model_info = {}
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
            elif hasattr(model, 'get_best_model_string'):
                model_info['formula'] = model.get_best_model_string()
            
            execution_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'execution_time': execution_time,
                'mse': mse,
                'r2': r2,
                'model_info': model_info,
                'error': None
            }
            
            logger.info(f"成功 测试 '{test_name}' 成功完成 (用时: {execution_time:.2f}s, R²: {r2:.4f})")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"失败 测试 '{test_name}' 失败: {error_msg}")
            logger.debug(traceback.format_exc())
            
            result = {
                'status': 'failed',
                'execution_time': execution_time,
                'mse': None,
                'r2': None,
                'model_info': {},
                'error': error_msg
            }
        
        self.test_results[test_name] = result
        return result
    
    def test_evolutionary_algorithms(self):
        """测试进化算法类方法"""
        logger.info("🧬 开始测试进化算法类方法")
        
        # 1. 遗传编程 (Genetic Programming)
        gp_params = {
            'population_size': 50,
            'n_generations': 10,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'max_depth': 6,
            'n_jobs': 1
        }
        self.run_single_test("遗传编程 (GP)", GeneticProgramming, gp_params)
        
        # 2. 遗传算法+粒子群优化混合
        ga_pso_params = {
            'population_size': 30,
            'generations': 10,
            'crossover_prob': 0.7,
            'mutation_prob': 0.1,
            'max_depth': 5
        }
        self.run_single_test("遗传算法+PSO混合", GAPSORegressor, ga_pso_params)
        
        # 3. 岛屿遗传编程 (暂时注释掉)
        island_gp_params = {
            'n_islands': 3,
            'island_size': 20,
            'generations': 8,
            'migration_freq': 5,
            'migration_size': 2
        }
        # self.run_single_test("岛屿遗传编程", IslandGPRegressor, island_gp_params)
    
    def test_sparse_modeling_methods(self):
        """测试稀疏建模与筛选类方法"""
        logger.info("开始测试稀疏建模与筛选类方法")
        
        # 1. SISSO 基础配置
        sisso_params = {
            'K': 2,
            'sis_topk': 50,
            'so_max_terms': 3,
            'sis_screener': 'pearson',
            'so_solver': 'omp',
            'cv': 3
        }
        self.run_single_test("SISSO (基础)", SISSORegressor, sisso_params)
        
        # 2. SISSO 不同筛选器
        for screener in ['pearson', 'f_regression', 'mutual_info']:
            params = {**sisso_params, 'sis_screener': screener}
            self.run_single_test(f"SISSO (筛选器: {screener})", SISSORegressor, params)
        
        # 3. SISSO 不同求解器
        for solver in ['omp', 'lasso', 'elasticnet']:
            params = {**sisso_params, 'so_solver': solver}
            self.run_single_test(f"SISSO (求解器: {solver})", SISSORegressor, params)
        
        # 4. SISSO 维度检查
        sisso_dim_params = {**sisso_params, 'dimensional_check': True}
        self.run_single_test("SISSO (维度检查)", SISSORegressor, sisso_dim_params, use_dimensions=True)
        
        # 5. LASSO稀疏回归
        sparse_params = {
            'alpha': 0.01,
            'max_iter': 1000,
            'poly_degree': 2,
            'normalize': True
        }
        self.run_single_test("LASSO稀疏回归", LassoRegressor, sparse_params)
        
        # 6. SINDy (Sparse Identification of Nonlinear Dynamics)
        sindy_params = {
            'threshold': 0.01,
            'alpha': 0.05,
            'poly_degree': 2,
            'solver': 'lasso'
        }
        self.run_single_test("SINDy", SINDyRegressor, sindy_params)
    
    def test_bayesian_probabilistic_methods(self):
        """测试贝叶斯与概率模型类方法"""
        logger.info("🎲 开始测试贝叶斯与概率模型类方法")
        
        # 1. 贝叶斯符号回归
        bsr_params = {
            'n_iter': 500,
            'n_chains': 2,
            'max_expr_depth': 5,
            'temperature': 1.0
        }
        self.run_single_test("贝叶斯符号回归 (MCMC)", BayesianSymbolicRegressor, bsr_params)
        
        # 2. 概率程序归纳
        ppi_params = {
            'n_iterations': 1000,
            'population_size': 50,
            'max_expr_depth': 5,
            'prior_temp': 1.0
        }
        self.run_single_test("概率程序归纳 (PCFG)", ProbabilisticProgramInduction, ppi_params)
    
    def test_reinforcement_neural_methods(self):
        """测试强化学习与神经符号方法"""
        logger.info("🤖 开始测试强化学习与神经符号方法")
        
        # 注意：这些方法可能需要额外的深度学习框架
        try:
            from sisso_py.neural_symbolic.rl_sr import ReinforcementSymbolicRegression
            from sisso_py.neural_symbolic.deep_sr import DeepSymbolicRegression
            from sisso_py.neural_symbolic.hybrid_neural import NeuralSymbolicHybrid
            
            # 1. 强化学习驱动的符号回归
            rl_params = {
                'agent_type': 'dqn',
                'max_episodes': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
            self.run_single_test("强化学习符号回归", ReinforcementSymbolicRegression, rl_params)
            
            # 2. 深度符号回归
            deep_sr_params = {
                'encoder_layers': [64, 32],
                'decoder_layers': [32, 64],
                'max_length': 20,
                'epochs': 50
            }
            self.run_single_test("深度符号回归", DeepSymbolicRegression, deep_sr_params)
            
            # 3. 神经-符号混合模型
            hybrid_params = {
                'neural_component': 'transformer',
                'symbolic_component': 'gp',
                'fusion_method': 'attention',
                'epochs': 30
            }
            self.run_single_test("神经符号混合", NeuralSymbolicHybrid, hybrid_params)
            
        except ImportError:
            logger.warning("⚠️ 神经符号方法模块未找到，跳过相关测试")
    
    def test_hybrid_emerging_methods(self):
        """测试混合与新兴方法"""
        logger.info("🔬 开始测试混合与新兴方法")
        
        try:
            from sisso_py.hybrid.evolutionary_gradient import EvolutionaryGradientHybrid
            from sisso_py.hybrid.physics_informed import PhysicsInformedSymbolicRegression
            from sisso_py.hybrid.multi_objective import MultiObjectiveSymbolicRegression
            
            # 1. 进化搜索 + 梯度优化混合
            evo_grad_params = {
                'evolution_phase_generations': 20,
                'gradient_phase_iterations': 100,
                'population_size': 30,
                'learning_rate': 0.01
            }
            self.run_single_test("进化+梯度混合", EvolutionaryGradientHybrid, evo_grad_params)
            
            # 2. 物理约束与维度分析
            physics_params = {
                'physical_constraints': ['conservation_laws'],
                'dimensional_analysis': True,
                'constraint_weight': 0.1,
                'K': 2
            }
            self.run_single_test("物理约束符号回归", PhysicsInformedSymbolicRegression, physics_params, use_dimensions=True)
            
            # 3. 多目标符号回归
            multi_obj_params = {
                'objectives': ['accuracy', 'complexity', 'interpretability'],
                'population_size': 50,
                'n_generations': 20,
                'pareto_front_size': 10
            }
            self.run_single_test("多目标符号回归", MultiObjectiveSymbolicRegression, multi_obj_params)
            
        except ImportError:
            logger.warning("⚠️ 混合方法模块未找到，跳过相关测试")
    
    def run_comprehensive_test(self):
        """运行全面测试"""
        logger.info("🚀 开始全面符号回归方法测试")
        
        # 生成测试数据
        logger.info("📊 生成测试数据...")
        self.generate_test_data(n_samples=80, complexity='medium')
        
        # 运行各类测试
        self.test_evolutionary_algorithms()
        self.test_sparse_modeling_methods()
        self.test_bayesian_probabilistic_methods()
        self.test_reinforcement_neural_methods()
        self.test_hybrid_emerging_methods()
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        logger.info("📋 生成测试报告...")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results.values() if r['status'] == 'success')
        failed_tests = total_tests - successful_tests
        
        print("\n" + "="*80)
        print("符号回归方法全面测试报告")
        print("="*80)
        print(f"总测试数: {total_tests}")
        print(f"成功测试: {successful_tests} ✅")
        print(f"失败测试: {failed_tests} ❌")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%")
        print("="*80)
        
        # 详细结果
        print("\n📊 详细测试结果:")
        print("-" * 80)
        
        categories = {
            "进化算法类": ["遗传编程", "遗传算法+PSO", "岛屿遗传编程"],
            "稀疏建模类": ["SISSO", "LASSO", "SINDy"],
            "贝叶斯概率类": ["贝叶斯符号回归", "概率程序归纳"],
            "强化学习类": ["强化学习符号回归", "深度符号回归", "神经符号混合"],
            "混合新兴类": ["进化+梯度混合", "物理约束符号回归", "多目标符号回归"]
        }
        
        for category, keywords in categories.items():
            print(f"\n🏷️ {category}:")
            category_tests = [name for name in self.test_results.keys() 
                            if any(keyword in name for keyword in keywords)]
            
            for test_name in category_tests:
                result = self.test_results[test_name]
                status_icon = "✅" if result['status'] == 'success' else "❌"
                time_str = f"{result['execution_time']:.2f}s"
                
                if result['status'] == 'success' and result['r2'] is not None:
                    r2_str = f"R²={result['r2']:.4f}"
                    print(f"  {status_icon} {test_name:<30} | {time_str:<8} | {r2_str}")
                else:
                    error_str = result['error'][:50] + "..." if result['error'] and len(result['error']) > 50 else result['error']
                    print(f"  {status_icon} {test_name:<30} | {time_str:<8} | {error_str}")
        
        # 性能排名
        successful_results = [(name, result) for name, result in self.test_results.items() 
                            if result['status'] == 'success' and result['r2'] is not None]
        
        if successful_results:
            print(f"\n🏆 性能排名 (按R²得分):")
            print("-" * 60)
            sorted_results = sorted(successful_results, key=lambda x: x[1]['r2'], reverse=True)
            
            for i, (name, result) in enumerate(sorted_results[:10], 1):
                print(f"  {i:2d}. {name:<35} R²={result['r2']:.4f}")
        
        print("\n" + "="*80)
        
        if failed_tests == 0:
            print("🎉 恭喜！所有符号回归方法测试都通过了！")
        else:
            print(f"⚠️ 有 {failed_tests} 个测试失败，请检查上述错误信息")
        
        print("="*80)


def main():
    """主函数"""
    print("🔬 符号回归方法全面测试系统")
    print("测试范围包括:")
    print("  1️⃣ 进化算法类 (GP, GEP, 演化算子)")
    print("  2️⃣ 稀疏建模与筛选类 (SISSO, LASSO, SINDy)")
    print("  3️⃣ 贝叶斯与概率模型类 (BSR, PCFG)")
    print("  4️⃣ 强化学习与神经符号方法 (RL-SR, Deep SR)")
    print("  5️⃣ 混合与新兴方法 (物理约束, 多目标)")
    print()
    
    # 创建测试套件并运行
    test_suite = SymbolicRegressionTestSuite()
    test_suite.run_comprehensive_test()


if __name__ == "__main__":
    main()
