#!/usr/bin/env python3
"""
Windows兼容性和CUDA可用性测试脚本
"""
import sys
import os
import platform
import importlib.util

def test_python_version():
    """测试Python版本"""
    print("🐍 Python版本检查...")
    version = sys.version_info
    print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python版本符合要求 (3.8+)")
        return True
    else:
        print("   ❌ Python版本过低，需要3.8或更高版本")
        return False

def test_platform():
    """测试平台信息"""
    print("\n🖥️ 平台信息...")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   架构: {platform.machine()}")
    print(f"   处理器: {platform.processor()}")
    
    if platform.system() == "Windows":
        print("   ✅ Windows平台检测成功")
        return True
    else:
        print(f"   ℹ️ 非Windows平台: {platform.system()}")
        return True

def test_pytorch_cuda():
    """测试PyTorch和CUDA"""
    print("\n🔥 PyTorch和CUDA检查...")
    
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            print("   ✅ CUDA环境就绪")
        else:
            print("   ⚠️ CUDA不可用，将使用CPU训练")
        
        return True
        
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def test_dependencies():
    """测试其他依赖"""
    print("\n📦 依赖包检查...")
    
    dependencies = [
        ('pygame', 'Pygame'),
        ('gym', 'OpenAI Gym'),
        ('numpy', 'NumPy'),
    ]
    
    all_available = True
    
    for module_name, display_name in dependencies:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                module = importlib.import_module(module_name)
                version = getattr(module, '__version__', 'Unknown')
                print(f"   ✅ {display_name}: {version}")
            else:
                print(f"   ❌ {display_name}: 未安装")
                all_available = False
        except ImportError:
            print(f"   ❌ {display_name}: 导入失败")
            all_available = False
    
    return all_available

def test_file_structure():
    """测试文件结构"""
    print("\n📁 项目结构检查...")
    
    required_files = [
        'src/game.py',
        'src/env.py',
        'src/env_improved.py',
        'train/train.py',
        'train/train_improved.py',
        'demo/demo.py',
        'requirements.txt',
        'requirements-cuda.txt',
    ]
    
    if platform.system() == "Windows":
        required_files.extend([
            'setup_windows.bat',
            'setup_cuda.bat',
            'train_windows.bat',
            'demo_windows.bat'
        ])
    else:
        required_files.append('setup.sh')
    
    all_present = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (缺失)")
            all_present = False
    
    return all_present

def test_model_directory():
    """测试模型目录"""
    print("\n🗂️ 模型目录检查...")
    
    models_dir = 'models'
    if os.path.exists(models_dir):
        print(f"   ✅ {models_dir} 目录存在")
        
        # 检查已有模型
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if model_files:
            print("   已有模型文件:")
            for model_file in model_files:
                file_size = os.path.getsize(os.path.join(models_dir, model_file)) / 1024
                print(f"     📄 {model_file} ({file_size:.1f}KB)")
        else:
            print("   📝 暂无模型文件（训练后生成）")
        
        return True
    else:
        print(f"   ❌ {models_dir} 目录不存在")
        return False

def main():
    """主测试函数"""
    print("🧪 Snake RL - Windows兼容性测试")
    print("=" * 50)
    
    tests = [
        ("Python版本", test_python_version),
        ("平台信息", test_platform),
        ("PyTorch和CUDA", test_pytorch_cuda),
        ("依赖包", test_dependencies),
        ("文件结构", test_file_structure),
        ("模型目录", test_model_directory),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置完成，可以开始使用。")
        
        if platform.system() == "Windows":
            print("\n💡 Windows用户快速开始:")
            print("   1. 训练模型: train_windows.bat")
            print("   2. 运行演示: demo_windows.bat")
        else:
            print("\n💡 Linux/Mac用户快速开始:")
            print("   1. 训练模型: python train/train_improved.py")
            print("   2. 运行演示: python demo_improved.py")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
        
        if not any(result for name, result in results if "依赖" in name):
            print("\n🔧 依赖安装建议:")
            if platform.system() == "Windows":
                print("   运行: setup_windows.bat 或 setup_cuda.bat")
            else:
                print("   运行: ./setup.sh")

if __name__ == "__main__":
    main()