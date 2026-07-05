"""
main.py —— 身份证多场景三分类总调度入口
=========================================
功能：
  1. 统一配置数据集根路径、所有超参数
  2. 按顺序调度全流程：
     预处理 → 模型训练调优 → 测试评估
  3. 异常处理、流程日志打印
  4. 断点复用：已存在的特征/模型文件自动跳过

用法：
  python main.py

  或修改下方 CONFIG 字典中的 dataset_root 后直接运行。

作者：计算机视觉工程
日期：2026-07-05
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
#  全局配置 —— 用户仅需修改此处！
# ============================================================================

CONFIG = {
    # 数据集根目录
    "dataset_root": "dataset/",
    
    # 输出目录（特征、模型、评估结果等）
    "output_dir": "outputs/",
    
    # 分类模式：False=三分类(实拍/拍屏/复印件), True=二分类(实拍 vs 非实拍)
    "binary_mode": False,
    
    # ===== 预处理参数 =====
    "skip_preprocess": False,     # True: 若特征文件已存在则跳过预处理
    "patch_size": 256,            # 切块大小
    "stride": 256,                # 切块步长（等于 patch_size → 无重叠）
    "laplacian_thresh": 15,       # 无纹理切块过滤阈值
    
    # ===== 训练参数 =====
    "skip_training": False,       # True: 若模型文件已存在则跳过训练
    
    # 网格搜索超参数空间
    "C_list": [0.1, 1, 10, 100],              # 惩罚参数 C 候选值
    "gamma_list": [0.001, 0.01, 0.1, 1],     # RBF 核 γ 候选值
    
    "cv_folds": 5,                # 交叉验证折数
    "smo_tol": 1e-3,             # KKT 条件容忍度
    "smo_max_iter": 2000,        # SMO 最大迭代次数
    
    # ===== 输出文件路径 =====
    "X_train_path": "outputs/X_train.pkl",
    "y_train_path": "outputs/y_train.pkl",
    "X_test_path": "outputs/X_test.pkl",
    "y_test_path": "outputs/y_test.pkl",
    "scaler_path": "outputs/scaler.pkl",
    "model_path": "outputs/best_model.pkl",
    "cv_results_path": "outputs/cv_results.pkl",
    "confusion_matrix_path": "outputs/confusion_matrix.png",
}


# ============================================================================
#  主入口函数
# ============================================================================

def main():
    """主调度流程"""
    
    parser = argparse.ArgumentParser(
        description="身份证多场景三分类 —— 总调度入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python main.py
  python main.py --dataset_root /path/to/dataset --output_dir results/
  python main.py --skip_preprocess --skip_training
        """,
    )
    parser.add_argument("--dataset_root", type=str, default=None, help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--skip_preprocess", action="store_true", default=None, help="跳过预处理")
    parser.add_argument("--skip_training", action="store_true", default=None, help="跳过训练")
    parser.add_argument("--C_list", type=float, nargs="+", default=None, help="C候选值列表")
    parser.add_argument("--gamma_list", type=float, nargs="+", default=None, help="gamma候选值列表")
    parser.add_argument("--max_iter", type=int, default=None, help="SMO最大迭代次数")
    parser.add_argument("--cv_folds", type=int, default=None, help="交叉验证折数")
    args = parser.parse_args()
    
    # 合并命令行参数到 CONFIG
    if args.dataset_root is not None:
        CONFIG["dataset_root"] = args.dataset_root
    if args.output_dir is not None:
        CONFIG["output_dir"] = args.output_dir
        # 同步更新所有输出路径
        CONFIG["X_train_path"] = os.path.join(args.output_dir, "X_train.pkl")
        CONFIG["y_train_path"] = os.path.join(args.output_dir, "y_train.pkl")
        CONFIG["X_test_path"] = os.path.join(args.output_dir, "X_test.pkl")
        CONFIG["y_test_path"] = os.path.join(args.output_dir, "y_test.pkl")
        CONFIG["scaler_path"] = os.path.join(args.output_dir, "scaler.pkl")
        CONFIG["model_path"] = os.path.join(args.output_dir, "best_model.pkl")
        CONFIG["cv_results_path"] = os.path.join(args.output_dir, "cv_results.pkl")
        CONFIG["confusion_matrix_path"] = os.path.join(args.output_dir, "confusion_matrix.png")
    if args.skip_preprocess:
        CONFIG["skip_preprocess"] = True
    if args.skip_training:
        CONFIG["skip_training"] = True
    if args.C_list is not None:
        CONFIG["C_list"] = list(args.C_list)
    if args.gamma_list is not None:
        CONFIG["gamma_list"] = list(args.gamma_list)
    if args.max_iter is not None:
        CONFIG["smo_max_iter"] = args.max_iter
    if args.cv_folds is not None:
        CONFIG["cv_folds"] = args.cv_folds
    
    # ================================================================
    #  打印启动信息
    # ================================================================
    print("\n" + "=" * 70)
    print("  身份证多场景三分类工程")
    print("  实拍 (Real) vs 拍屏 (Screen) vs 复印件翻拍 (Paper)")
    print("  手写 RBF 核 SVM + LBP/HOG/HSV 手工特征")
    print("=" * 70)
    binary_mode = CONFIG.get("binary_mode", False)
    print(f"\n  配置信息:")
    print(f"    数据集路径:       {CONFIG['dataset_root']}")
    print(f"    输出目录:         {CONFIG['output_dir']}")
    print(f"    分类模式:         {'二分类（实拍 vs 非实拍）' if binary_mode else '三分类（实拍/拍屏/复印件）'}")
    print(f"    C 候选:           {CONFIG['C_list']}")
    print(f"    gamma 候选:       {CONFIG['gamma_list']}")
    print(f"    交叉验证折数:     {CONFIG['cv_folds']}")
    print(f"    SMO 最大迭代:     {CONFIG['smo_max_iter']}")
    print(f"    跳过预处理:       {CONFIG['skip_preprocess']}")
    print(f"    跳过训练:         {CONFIG['skip_training']}")
    print()
    
    total_start = time.time()
    
    try:
        # ============================================================
        #  Stage 1: 预处理
        # ============================================================
        print("\n" + "▸" * 35)
        print("  Stage 1/3: 数据预处理")
        print("▸" * 35 + "\n")
        
        from pre_process import pre_process_pipeline
        
        # 注入切块参数到 pre_process 模块
        import pre_process as pp_module
        pp_module.PATCH_SIZE = CONFIG["patch_size"]
        pp_module.STRIDE = CONFIG["stride"]
        pp_module.LAPLACIAN_THRESH = CONFIG["laplacian_thresh"]
        
        feat_paths = pre_process_pipeline(
            dataset_root=CONFIG["dataset_root"],
            output_dir=CONFIG["output_dir"],
            skip_if_exists=CONFIG["skip_preprocess"],
            binary_mode=binary_mode,
        )
        
        # 验证预处理输出
        required_files = [
            CONFIG["X_train_path"],
            CONFIG["y_train_path"],
            CONFIG["X_test_path"],
            CONFIG["y_test_path"],
            CONFIG["scaler_path"],
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"预处理未生成以下文件: {missing}")
        
        print(f"\n[主流程] ✓ Stage 1/3 预处理完成")
        
        # ============================================================
        #  Stage 2: 训练
        # ============================================================
        print("\n" + "▸" * 35)
        print("  Stage 2/3: 模型训练与调优")
        print("▸" * 35 + "\n")
        
        from train import train_pipeline
        
        model, best_cv_acc, cv_results = train_pipeline(
            features_dir=CONFIG["output_dir"],
            model_path=CONFIG["model_path"],
            cv_results_path=CONFIG["cv_results_path"],
            skip_if_exists=CONFIG["skip_training"],
            binary_mode=binary_mode,
            C_list=CONFIG["C_list"],
            gamma_list=CONFIG["gamma_list"],
            n_splits=CONFIG["cv_folds"],
            tol=CONFIG["smo_tol"],
            max_iter=CONFIG["smo_max_iter"],
        )
        
        if not os.path.exists(CONFIG["model_path"]):
            raise FileNotFoundError(f"训练未生成模型文件: {CONFIG['model_path']}")
        
        print(f"\n[主流程] ✓ Stage 2/3 训练完成")
        print(f"  最优参数: C={model.C}, gamma={model.gamma}")
        print(f"  交叉验证准确率: {best_cv_acc:.4f}")
        
        # ============================================================
        #  Stage 3: 测试评估
        # ============================================================
        print("\n" + "▸" * 35)
        print("  Stage 3/3: 测试评估")
        print("▸" * 35 + "\n")
        
        from test import test_pipeline
        
        metrics = test_pipeline(
            features_dir=CONFIG["output_dir"],
            model_path=CONFIG["model_path"],
            confusion_matrix_path=CONFIG["confusion_matrix_path"],
            binary_mode=binary_mode,
        )
        
        print(f"\n[主流程] ✓ Stage 3/3 测试评估完成")
        
        # ============================================================
        #  最终汇总
        # ============================================================
        total_time = time.time() - total_start
        
        print("\n" + "=" * 70)
        print("  全流程完成!")
        print("=" * 70)
        print(f"  总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"  测试集准确率: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"  混淆矩阵图片: {CONFIG['confusion_matrix_path']}")
        print(f"  最优模型文件: {CONFIG['model_path']}")
        print(f"  特征文件目录: {CONFIG['output_dir']}")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[错误] 文件未找到: {e}")
        print("  可能原因:")
        print("    1. 数据集路径不正确，请检查 --dataset_root 参数")
        print("    2. 预处理未成功执行，请先运行预处理")
        print("    3. 输出目录权限不足")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\n[错误] 导入模块失败: {e}")
        print("  请确认在 conda mchlrn 环境下运行:")
        print("    conda activate mchlrn")
        print("  并确保已安装: numpy, opencv-python, scikit-image, scikit-learn, matplotlib, seaborn, tqdm")
        sys.exit(1)
        
    except MemoryError as e:
        print(f"\n[错误] 内存不足: {e}")
        print("  建议:")
        print("    1. 减小图片分辨率或切块大小")
        print("    2. 减小训练样本数")
        print("    3. 增加系统 swap 空间")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[错误] 未预期的异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()