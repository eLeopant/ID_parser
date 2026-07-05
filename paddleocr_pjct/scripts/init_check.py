#!/usr/bin/env python3
"""
初始化检查脚本 - 测试 PaddleOCR 是否能正常导入并打印版本号
"""

def check_paddleocr():
    try:
        from paddleocr import PaddleOCR
        # 获取 paddleocr 版本号
        import paddleocr
        print(f"✅ PaddleOCR 导入成功！")
        print(f"   PaddleOCR 版本: {paddleocr.__version__}")
        
        # 检查 PaddlePaddle 版本
        import paddle
        print(f"   PaddlePaddle 版本: {paddle.__version__}")
        if paddle.is_compiled_with_cuda():
            print(f"   GPU 可用: ✅ (CUDA 版本: {paddle.version.cuda()})")
        else:
            print(f"   GPU 可用: ❌ (使用 CPU 模式)")
        
        return True
    except ImportError as e:
        print(f"❌ PaddleOCR 导入失败: {e}")
        print("请执行: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  PaddleOCR 初始化环境检查")
    print("=" * 50)
    success = check_paddleocr()
    print("=" * 50)
    if success:
        print("  ✅ 环境检查通过，可以开始开发！")
    else:
        print("  ❌ 环境检查未通过，请检查依赖安装。")
    print("=" * 50)
