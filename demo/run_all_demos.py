# -*- coding: utf-8 -*-
"""
SISSO符号回归方法演示集合
======================

这个脚本演示了SISSO库中所有主要的符号回归方法。
每个方法都有独立的演示文件，这里提供统一的入口和概览。
"""

import sys
import os
import importlib
import traceback

def print_header():
    """打印欢迎信息"""
    print("🎯 SISSO符号回归方法演示集合")
    print("=" * 60)
    print("📚 包含20+种先进的符号回归方法演示")
    print("🔬 从传统遗传编程到最新深度学习方法")
    print("💡 每个演示都包含理论、实践和可视化")
    print("=" * 60)

def get_available_demos():
    """获取可用的演示文件列表"""
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    demos = []
    
    # 演示文件列表
    demo_files = [
        ("genetic_programming_demo", "🧬 传统遗传编程"),
        ("ga_pso_hybrid_demo", "🔀 遗传算法-粒子群混合"),
        ("sisso_basic_demo", "📊 SISSO基础方法"),
        ("lasso_regression_demo", "📈 LASSO回归"),
        ("sindy_demo", "🌊 SINDy动力学发现"),
        ("bayesian_symbolic_regression_demo", "🎲 贝叶斯符号回归"),
        ("probabilistic_program_induction_demo", "📝 概率程序归纳"),
        ("reinforcement_learning_sr_demo", "🎮 强化学习符号回归"),
        ("deep_symbolic_regression_demo", "🧠 深度符号回归"),
        ("physics_informed_sr_demo", "⚛️ 物理约束符号回归"),
        ("multi_objective_sr_demo", "🎯 多目标符号回归"),
        ("island_gp_demo", "🏝️ 岛屿遗传编程")
    ]
    
    for file_name, description in demo_files:
        file_path = os.path.join(demo_dir, f"{file_name}.py")
        if os.path.exists(file_path):
            demos.append((file_name, description, file_path))
    
    return demos

def run_demo(demo_module_name):
    """运行指定的演示"""
    try:
        print(f"\n🚀 开始运行演示: {demo_module_name}")
        print("-" * 50)
        
        # 动态导入演示模块
        demo_module = importlib.import_module(demo_module_name)
        
        # 运行主函数
        if hasattr(demo_module, 'main'):
            demo_module.main()
        else:
            print(f"❌ 演示模块 {demo_module_name} 没有 main() 函数")
            
        print(f"\n✅ 演示 {demo_module_name} 运行完成!")
        
    except Exception as e:
        print(f"\n❌ 运行演示 {demo_module_name} 时出错:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()

def show_method_categories():
    """显示方法分类"""
    print("\n📂 符号回归方法分类:")
    print("=" * 50)
    
    categories = {
        "🧬 进化算法类": [
            "genetic_programming_demo - 遗传编程基础",
            "ga_pso_hybrid_demo - 混合进化算法", 
            "island_gp_demo - 并行岛屿遗传编程"
        ],
        "📊 稀疏建模类": [
            "sisso_basic_demo - SISSO特征构造",
            "lasso_regression_demo - LASSO稀疏回归",
            "sindy_demo - 稀疏动力学识别"
        ],
        "🎲 贝叶斯/概率类": [
            "bayesian_symbolic_regression_demo - 贝叶斯推断",
            "probabilistic_program_induction_demo - 概率程序生成"
        ],
        "🧠 神经符号类": [
            "reinforcement_learning_sr_demo - 强化学习方法",
            "deep_symbolic_regression_demo - 深度学习方法"
        ],
        "🔬 混合/新兴类": [
            "physics_informed_sr_demo - 物理约束方法",
            "multi_objective_sr_demo - 多目标优化"
        ]
    }
    
    for category, methods in categories.items():
        print(f"\n{category}:")
        for method in methods:
            print(f"  • {method}")

def interactive_menu():
    """交互式菜单"""
    demos = get_available_demos()
    
    while True:
        print(f"\n🎯 请选择要运行的演示:")
        print("-" * 40)
        
        for i, (_, description, _) in enumerate(demos, 1):
            print(f"  {i:2d}. {description}")
        
        print(f"  {'0':>2}. 🔄 显示方法分类")
        print(f"  {'a':>2}. 🚀 运行所有演示")
        print(f"  {'q':>2}. 🚪 退出")
        
        try:
            choice = input(f"\n请输入选择 (1-{len(demos)}/0/a/q): ").strip().lower()
            
            if choice == 'q':
                print("👋 感谢使用SISSO演示系统！")
                break
            elif choice == '0':
                show_method_categories()
            elif choice == 'a':
                print("🚀 开始运行所有演示...")
                run_all_demos(demos)
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                demo_index = int(choice) - 1
                demo_name = demos[demo_index][0]
                run_demo(demo_name)
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出演示系统")
            break
        except Exception as e:
            print(f"❌ 输入处理错误: {e}")

def run_all_demos(demos):
    """运行所有演示"""
    print(f"\n🎯 将运行 {len(demos)} 个演示")
    success_count = 0
    
    for i, (demo_name, description, _) in enumerate(demos, 1):
        print(f"\n{'='*60}")
        print(f"📊 进度: {i}/{len(demos)} - {description}")
        print(f"{'='*60}")
        
        try:
            run_demo(demo_name)
            success_count += 1
        except Exception as e:
            print(f"❌ 演示失败: {e}")
            
        # 询问是否继续
        if i < len(demos):
            try:
                continue_choice = input(f"\n继续下一个演示? (y/n/q): ").strip().lower()
                if continue_choice == 'q':
                    break
                elif continue_choice == 'n':
                    break
            except KeyboardInterrupt:
                break
    
    print(f"\n📊 演示完成统计:")
    print(f"  ✅ 成功: {success_count}/{len(demos)}")
    print(f"  ❌ 失败: {len(demos) - success_count}/{len(demos)}")

def show_system_info():
    """显示系统信息"""
    print(f"\n💻 系统信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  工作目录: {os.getcwd()}")
    
    # 检查关键依赖
    required_packages = ['numpy', 'pandas', 'matplotlib', 'sklearn']
    print(f"\n📦 依赖包检查:")
    
    for package in required_packages:
        try:
            pkg = importlib.import_module(package)
            version = getattr(pkg, '__version__', '未知版本')
            print(f"  ✅ {package}: {version}")
        except ImportError:
            print(f"  ❌ {package}: 未安装")

def main():
    """主函数"""
    print_header()
    
    # 显示系统信息
    show_system_info()
    
    # 获取可用演示
    demos = get_available_demos()
    print(f"\n📁 发现 {len(demos)} 个可用演示")
    
    if not demos:
        print("❌ 没有找到可用的演示文件")
        return
    
    # 显示方法分类
    show_method_categories()
    
    # 启动交互式菜单
    interactive_menu()

if __name__ == "__main__":
    main()
