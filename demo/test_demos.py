# -*- coding: utf-8 -*-
"""
演示文件测试脚本
==============

测试所有演示文件是否可以正常导入和基本运行。
"""

import os
import sys
import importlib.util

def test_demo_imports():
    """测试演示文件导入"""
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    
    demo_files = [
        "genetic_programming_demo.py",
        "ga_pso_hybrid_demo.py", 
        "sisso_basic_demo.py",
        "lasso_regression_demo.py",
        "sindy_demo.py",
        "bayesian_symbolic_regression_demo.py",
        "probabilistic_program_induction_demo.py",
        "reinforcement_learning_sr_demo.py",
        "deep_symbolic_regression_demo.py",
        "physics_informed_sr_demo.py",
        "multi_objective_sr_demo.py",
        "island_gp_demo.py"
    ]
    
    print("🧪 测试演示文件导入")
    print("=" * 50)
    
    success_count = 0
    total_count = len(demo_files)
    
    for demo_file in demo_files:
        file_path = os.path.join(demo_dir, demo_file)
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"❌ {demo_file}: 文件不存在")
                continue
            
            # 尝试导入模块
            module_name = demo_file[:-3]  # 移除.py扩展名
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                print(f"❌ {demo_file}: 无法创建模块规范")
                continue
                
            module = importlib.util.module_from_spec(spec)
            
            # 执行模块导入
            spec.loader.exec_module(module)
            
            # 检查是否有main函数
            if hasattr(module, 'main'):
                print(f"✅ {demo_file}: 导入成功，包含main函数")
            else:
                print(f"⚠️ {demo_file}: 导入成功，但缺少main函数")
            
            success_count += 1
            
        except ImportError as e:
            print(f"❌ {demo_file}: 导入错误 - {str(e)}")
        except SyntaxError as e:
            print(f"❌ {demo_file}: 语法错误 - {str(e)}")
        except Exception as e:
            print(f"❌ {demo_file}: 其他错误 - {str(e)}")
    
    print(f"\n📊 测试结果:")
    print(f"  总文件数: {total_count}")
    print(f"  成功导入: {success_count}")
    print(f"  失败数量: {total_count - success_count}")
    print(f"  成功率: {success_count/total_count*100:.1f}%")
    
    return success_count == total_count

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包")
    print("=" * 30)
    
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'sklearn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺失的包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ 所有依赖包都已安装")
        return True

def main():
    """主函数"""
    print("🔍 SISSO演示文件测试")
    print("=" * 50)
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 测试导入
    imports_ok = test_demo_imports()
    
    print(f"\n🎯 总体结果:")
    if deps_ok and imports_ok:
        print("✅ 所有测试通过，演示系统就绪！")
        return True
    else:
        print("❌ 存在问题，请检查上述错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
