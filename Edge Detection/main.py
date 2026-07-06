"""
卡片边缘检测系统 - 主程序入口

功能:
- 方法1: KNN机器学习方法 (需要训练)
- 方法2: OpenCV传统方法 (无需训练，直接使用)

使用方法:
1. 将测试图片放入 data/test/images/
2. 运行 python main.py
3. 按照菜单提示进行操作
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 导入检测器
from knn_card_detection import CardEdgeDetector
from opencv_edge_detector import OpenCVEdgeDetector


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印标题"""
    print("\n" + "=" * 60)
    print("卡片边缘检测系统")
    print("       Card Edge Detection System")
    print("=" * 60)


def print_menu():
    """打印菜单"""
    print("\n请选择操作:")
    print("  1.训练KNN模型")
    print("  2. KNN方法测试")
    print("  3. OpenCV方法测试 (推荐，无需训练)")
    print("  4. 实时检测单张图片")
    print("  5. 查看数据目录结构")
    print("  6. 帮助说明")
    print("  0. 退出程序")
    print("-" * 60)


def show_directory_structure():
    """显示目录结构"""
    print("\n" + "=" * 60)
    print("数据目录结构")
    print("=" * 60)

    print("""
项目目录:
Edge Detection/
├── data/
│   ├── train/
│   │   ├── images/      ← 【训练集图片】放置原始卡片图片
│   │   └── labels/      ← 【训练集标签】放置角点坐标文件
│   ├── test/
│   │   ├── images/      ← 【测试集图片】放置待检测卡片图片
│   │   └── labels/      ← 【测试集标签】（可选）用于计算准确率
│   ├── models/          ← 训练好的模型保存位置
│   └── output/          ← 测试结果输出目录
├── knn_card_detection.py    ← KNN检测模块（需要训练）
├── opencv_edge_detector.py  ← OpenCV检测模块（无需训练）
├── main.py                ← 主程序入口
├── label_tool.py          ← 标签标注工具
└── requirements.txt       ← 依赖包
""")

    # 显示现有数据统计
    train_images = list(Path('data/train/images').glob('*.*'))
    train_labels = list(Path('data/train/labels').glob('*.txt'))
    test_images = list(Path('data/test/images').glob('*.*'))
    test_labels = list(Path('data/test/labels').glob('*.txt'))

    print("当前数据统计:")
    print(f"  训练图片: {len(train_images)} 张")
    print(f"  训练标签: {len(train_labels)} 个")
    print(f"  测试图片: {len(test_images)} 张")
    print(f"  测试标签: {len(test_labels)} 个")

    model_exists = Path('data/models/knn_card_model.pkl').exists()
    print(f"  KNN模型状态: {'已训练' if model_exists else '未训练'}")


def show_help():
    """显示帮助说明"""
    print("\n" + "=" * 60)
    print("帮助说明")
    print("=" * 60)
    print("""
【两种检测方法】

1. KNN方法 (机器学习)
   - 需要准备训练数据和标签进行训练
   - 训练后可以检测相似卡片
   - 适合批量处理相同类型的卡片

2. OpenCV方法 (传统图像处理) - 推荐
   - 无需训练，直接使用
   - 基于边缘检测和轮廓查找
   - 适合任何有明显边缘的卡片
   - 鲁棒性更好

【使用方法】

OpenCV方法 (推荐):
  运行程序选择 "3. OpenCV方法测试"

KNN方法:
  1. 将训练图片放入: data/train/images/
  2. 将标签文件放入: data/train/labels/
  3. 运行程序选择 "1. 训练KNN模型"
  4. 选择 "2. KNN方法测试"

【标签文件格式】
每行格式: x1,y1 x2,y2 x3,y3 x4,y4
示例: 0.1,0.1 0.9,0.1 0.9,0.9 0.1,0.9
""")


def train_knn_model():
    """训练KNN模型"""
    clear_screen()
    print_header()

    train_images = Path('data/train/images')
    train_labels = Path('data/train/labels')

    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(train_images.glob('*.jpg')) + list(train_images.glob('*.png'))

    if len(image_files) == 0:
        print("\n错误: 训练目录中没有找到图片!")
        print(f"请将训练图片放入: {train_images.absolute()}")
        input("\n按回车键返回菜单...")
        return

    label_files = list(train_labels.glob('*.txt'))
    if len(label_files) == 0:
        print("\n警告: 训练标签目录为空!")
        print(f"请将标签文件放入: {train_labels.absolute()}")

    print(f"\n找到 {len(image_files)} 张训练图片")
    print(f"找到 {len(label_files)} 个标签文件")

    k_value = input("\n请输入K值 (邻居数量，默认为3): ").strip()
    k_value = int(k_value) if k_value else 3

    confirm = input("\n确认开始训练? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消训练")
        input("\n按回车键返回菜单...")
        return

    try:
        detector = CardEdgeDetector()
        mse, mae, accuracy = detector.train(
            train_images_dir='data/train/images',
            train_labels_dir='data/train/labels',
            n_neighbors=k_value
        )
        print("\n训练完成!")
        print(f"模型已保存到: data/models/knn_card_model.pkl")

    except Exception as e:
        print(f"\n训练出错: {e}")

    input("\n按回车键返回菜单...")


def test_knn():
    """KNN方法测试"""
    clear_screen()
    print_header()

    detector = CardEdgeDetector()

    model_path = Path('data/models/knn_card_model.pkl')
    if model_path.exists():
        print("加载已训练的模型...")
        detector.load_model()
    else:
        print("\n错误: KNN模型未训练!")
        print("请先进行模型训练")
        input("\n按回车键返回菜单...")
        return

    test_images = Path('data/test/images')
    test_images.mkdir(parents=True, exist_ok=True)

    image_files = list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))

    if len(image_files) == 0:
        print("\n错误: 测试目录中没有找到图片!")
        print(f"请将测试图片放入: {test_images.absolute()}")
        input("\n按回车键返回菜单...")
        return

    print(f"\n找到 {len(image_files)} 张测试图片")

    confirm = input("\n确认开始测试? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消测试")
        input("\n按回车键返回菜单...")
        return

    try:
        results = detector.test(
            test_images_dir='data/test/images',
            test_labels_dir='data/test/labels',
            show_results=True
        )
        print("\n测试完成!")

    except Exception as e:
        print(f"\n测试出错: {e}")

    input("\n按回车键返回菜单...")


def test_opencv():
    """OpenCV方法测试"""
    clear_screen()
    print_header()

    print("使用OpenCV传统方法进行卡片边缘检测")
    print("此方法无需训练，直接使用边缘检测和轮廓查找")
    print()

    test_images = Path('data/test/images')
    test_images.mkdir(parents=True, exist_ok=True)

    image_files = list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))

    if len(image_files) == 0:
        print("\n错误: 测试目录中没有找到图片!")
        print(f"请将测试图片放入: {test_images.absolute()}")
        input("\n按回车键返回菜单...")
        return

    print(f"找到 {len(image_files)} 张测试图片")

    confirm = input("\n确认开始检测? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消检测")
        input("\n按回车键返回菜单...")
        return

    try:
        detector = OpenCVEdgeDetector()
        results = detector.test(
            test_images_dir='data/test/images',
            test_labels_dir='data/test/labels',
            show_results=True
        )
        print("\n检测完成!")

    except Exception as e:
        print(f"\n检测出错: {e}")

    input("\n按回车键返回菜单...")


def detect_single_image():
    """检测单张图片"""
    clear_screen()
    print_header()

    print("\n选择检测方法:")
    print("  1. KNN方法 (需要先训练模型)")
    print("  2. OpenCV方法 (推荐)")
    print("  0. 返回")

    choice = input("\n请输入选项: ").strip()

    image_path = input("请输入图片路径: ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"\n错误: 文件不存在: {image_path}")
        input("\n按回车键返回菜单...")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n错误: 无法读取图片: {image_path}")
        input("\n按回车键返回菜单...")
        return

    try:
        if choice == '1':
            detector = CardEdgeDetector()
            model_path = Path('data/models/knn_card_model.pkl')
            if not model_path.exists():
                print("\n错误: KNN模型未训练!")
                return
            detector.load_model()
            coords = detector.detect_edge(image)
            result_image = detector.draw_edge(image.copy(), coords)
            cropped_image = detector.crop_card(image, coords)
            print("\n使用KNN方法检测")

        else:
            detector = OpenCVEdgeDetector()
            coords = detector.detect_edge(image)
            result_image = detector.draw_edge(image.copy(), coords)
            cropped_image = detector.crop_card(image, coords)
            print("\n使用OpenCV方法检测")

        print(f"\n检测到的角点坐标 (归一化):")
        print(f"  左上: ({coords[0]:.4f}, {coords[1]:.4f})")
        print(f"  右上: ({coords[2]:.4f}, {coords[3]:.4f})")
        print(f"  右下: ({coords[4]:.4f}, {coords[5]:.4f})")
        print(f"  左下: ({coords[6]:.4f}, {coords[7]:.4f})")

        # 保存图片
        output_dir = Path('data/test/output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存带边框图片
        cv2.imwrite(str(output_dir / 'detection_result.png'), result_image)
        # 保存裁剪后的卡片
        cv2.imwrite(str(output_dir / 'cropped_card.png'), cropped_image)
        print(f"\n图片已保存到: {output_dir}")

        # 显示结果
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(result_rgb)
        axes[0].set_title('检测结果')
        axes[0].axis('off')

        axes[1].imshow(cropped_rgb)
        axes[1].set_title('裁剪后的卡片')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n检测出错: {e}")

    input("\n按回车键返回菜单...")


def main():
    """主函数"""
    clear_screen()
    print_header()

    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("请输入选项 (0-6): ").strip()

        if choice == '1':
            train_knn_model()
        elif choice == '2':
            test_knn()
        elif choice == '3':
            test_opencv()
        elif choice == '4':
            detect_single_image()
        elif choice == '5':
            clear_screen()
            show_directory_structure()
            input("\n按回车键返回菜单...")
        elif choice == '6':
            clear_screen()
            show_help()
            input("\n按回车键返回菜单...")
        elif choice == '0':
            print("\n感谢使用!再见!")
            break
        else:
            print("\n无效选项，请重新输入")
            input("\n按回车键继续...")


if __name__ == '__main__':
    main()