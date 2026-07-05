"""
test.py —— 身份证多场景三分类测试评估模块
=============================================
功能：
  1. 加载训练好的最优手写 SVM 模型（OvO多分类器 / 二分类）
  2. 加载标准化器与测试集特征/标签
  3. 模型推理预测，输出完整多分类/二分类评估指标：
     - 整体准确率（Accuracy）
     - 分类报告（Classification Report）
     - 混淆矩阵（可视化为热力图并保存为 PNG）
  4. 所有指标打印到控制台

评估指标使用 sklearn 计算（仅此用途，SVM 核心 100% 手写）。

作者：计算机视觉工程
日期：2026-07-05
"""

import os
import pickle
import argparse
import warnings
from typing import Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免无 GUI 时崩溃
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# 设置中文字体（支持中文标签显示）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 类别名称映射
CLASS_NAMES = {
    0: "实拍 (Real)",
    1: "拍屏 (Screen)",
    2: "复印件 (Paper)",
}

BINARY_CLASS_NAMES = {
    0: "实拍 (Real)",
    1: "非实拍 (Non-Real)",
}


# ============================================================================
#  加载资产
# ============================================================================

def _load_binary_model(filepath: str):
    """
    加载二分类 HandwrittenSVM 模型（mode='binary' 格式）。
    
    独立实现，避免与 train.py 中的同名函数代码重复问题；
    两者功能完全一致。
    """
    from train import HandwrittenSVM
    with open(filepath, "rb") as f:
        d = pickle.load(f)
    svm = HandwrittenSVM(C=d["C"], gamma=d["gamma"], tol=d["tol"], max_iter=d["max_iter"])
    svm.alpha = d["alpha"]
    svm.b = d["b"]
    svm.support_vectors = d["support_vectors"]
    svm.support_labels = d["support_labels"]
    svm.support_alpha = d["support_alpha"]
    svm.X_train = d["X_train"]
    return svm


def load_assets(
    features_dir: str,
    model_path: str,
    binary_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """
    加载测试所需的全部资产：特征、标签、标准化器、模型。
    
    Args:
        features_dir: 特征文件目录
        model_path: 模型 pkl 文件路径
        binary_mode: True=二分类模式，False=三分类模式
    
    Returns:
        (X_test, y_test, scaler, model)
    """
    from train import OvOSVMClassifier
    
    print("[评估] 加载测试资产...")
    
    # 加载测试特征和标签
    with open(os.path.join(features_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(features_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    
    # 加载标准化器
    with open(os.path.join(features_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    # 加载模型（自动检测二分类 vs OvO）
    with open(model_path, "rb") as f:
        header = pickle.load(f)
    if isinstance(header, dict) and header.get("mode") == "binary":
        model = _load_binary_model(model_path)
        print(f"  [二分类模式] 模型类型: HandwrittenSVM")
    else:
        model = OvOSVMClassifier.load(model_path)
        print(f"  [三分类模式] 模型类型: OvOSVMClassifier")
    
    print(f"  测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 维特征")
    if binary_mode:
        for cls_id, cls_name in sorted(BINARY_CLASS_NAMES.items()):
            cnt = np.sum(y_test == cls_id)
            print(f"    {cls_name}: {cnt} 样本")
    else:
        for cls_id, cls_name in sorted(CLASS_NAMES.items()):
            cnt = np.sum(y_test == cls_id)
            print(f"    {cls_name}: {cnt} 样本")
    
    return X_test, y_test, scaler, model


# ============================================================================
#  评估指标计算与输出
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
) -> Dict:
    """
    计算并打印所有分类评估指标。
    
    兼容二分类和三分类场景。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: {标签ID: "名称"} 映射
    
    Returns:
        包含所有指标的字典
    """
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("  测试集评估结果")
    print("=" * 60)
    print(f"\n  ✓ 整体准确率 (Accuracy): {acc:.4f} ({acc * 100:.2f}%)")
    
    # 分类报告
    print(f"\n  {'─' * 50}")
    print("  分类报告 (Classification Report):")
    print(f"  {'─' * 50}")
    
    target_names = [class_names.get(i, f"Class {i}") 
                    for i in sorted(class_names.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    n = len(target_names)
    
    print(f"  {'─' * 50}")
    print("  混淆矩阵 (Confusion Matrix):")
    print(f"  {'─' * 50}")
    
    # 动态打印表头
    header_col = "{:>18s}".format("真实\\预测")
    pred_headers = "".join(f"  预测{name[:6]:>6s}" for name in target_names)
    print(f"  {header_col}{pred_headers}")
    
    for i, name in enumerate(target_names):
        row = f"  真实{name:>12s}"
        for j in range(n):
            row += f"  {cm[i][j]:>8d}"
        print(row)
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict[int, str],
    save_path: str = "outputs/confusion_matrix.png",
    title: str = None,
) -> str:
    """
    绘制混淆矩阵热力图并保存为 PNG 图片。
    
    使用 seaborn 热力图，标注类别名称与百分比，清晰展示误诊分布。
    
    Args:
        cm: 混淆矩阵 (n_classes, n_classes)
        class_names: {标签ID: "名称"} 映射
        save_path: 图片保存路径
        title: 图表标题，None 则自动生成
    
    Returns:
        保存的图片路径
    """
    target_names = [class_names.get(i, f"Class {i}") 
                    for i in sorted(class_names.keys())]
    n_classes = len(target_names)
    
    # 计算百分比（行归一化）
    cm_pct = cm.astype(np.float64)
    for i in range(n_classes):
        row_sum = cm_pct[i].sum()
        if row_sum > 0:
            cm_pct[i] = cm_pct[i] / row_sum * 100
    
    # 创建标注文本（数量 + 百分比）
    annot = np.empty_like(cm, dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(7 if n_classes == 2 else 8, 5 if n_classes == 2 else 6))
    
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "百分比 (%)"},
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    
    ax.set_xlabel("预测标签", fontsize=13, fontweight="bold")
    ax.set_ylabel("真实标签", fontsize=13, fontweight="bold")
    
    if title is None:
        title = "身份证多场景分类 —— 混淆矩阵"
        if n_classes == 2:
            title = "身份证二分类（实拍 vs 非实拍）—— 混淆矩阵"
        else:
            title = "身份证多场景三分类 —— 混淆矩阵"
    
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n  ✓ 混淆矩阵热力图已保存至: {save_path}")
    return save_path


# ============================================================================
#  主评估管道
# ============================================================================

def test_pipeline(
    features_dir: str = "outputs/",
    model_path: str = "outputs/best_model.pkl",
    confusion_matrix_path: str = "outputs/confusion_matrix.png",
    binary_mode: bool = False,
) -> Dict:
    """
    完整测试评估管道：加载资产 → 推理 → 评估指标 → 混淆矩阵可视化。
    
    Args:
        features_dir: 特征文件目录
        model_path: 模型文件路径
        confusion_matrix_path: 混淆矩阵图片保存路径
        binary_mode: True=二分类模式，False=三分类模式
    
    Returns:
        包含所有评估指标的字典
    """
    # Step 1: 加载资产
    X_test, y_test, scaler, model = load_assets(
        features_dir, model_path, binary_mode=binary_mode
    )
    
    # Step 2: 模型推理
    print("\n[评估] 执行模型推理...")
    if binary_mode:
        # 二分类：HandwrittenSVM 返回 ±1，转换回 0/1
        y_pred_bin = model.predict(X_test)  # +1 或 -1
        y_pred = np.where(y_pred_bin == -1, 0, 1).astype(np.int32)
    else:
        y_pred = model.predict(X_test)
    
    # Step 3: 计算并输出指标
    if binary_mode:
        class_names = BINARY_CLASS_NAMES
        title = "身份证二分类（实拍 vs 非实拍）—— 混淆矩阵"
    else:
        class_names = CLASS_NAMES
        title = "身份证多场景三分类 —— 混淆矩阵"
    
    metrics = compute_metrics(y_test, y_pred, class_names)
    
    # Step 4: 混淆矩阵可视化
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        confusion_matrix_path, title=title,
    )
    
    # Step 5: 额外分析——逐类准确率
    print(f"\n  {'─' * 50}")
    print("  逐类准确率:")
    print(f"  {'─' * 50}")
    for cls_id, cls_name in sorted(class_names.items()):
        mask = y_test == cls_id
        if mask.sum() > 0:
            cls_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"    {cls_name:>18s}: {cls_acc:.4f} ({cls_acc * 100:.2f}%)")
        else:
            print(f"    {cls_name:>18s}: 无样本")
    
    print(f"\n{'=' * 60}")
    print(f"  测试评估完成")
    print(f"{'=' * 60}\n")
    
    return metrics


# ============================================================================
#  CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="身份证分类测试评估模块")
    parser.add_argument("--features_dir", type=str, default="outputs/", 
                        help="特征文件目录")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pkl", 
                        help="模型文件路径")
    parser.add_argument("--confusion_matrix_path", type=str, 
                        default="outputs/confusion_matrix.png", 
                        help="混淆矩阵保存路径")
    parser.add_argument("--binary", action="store_true", default=False,
                        help="二分类模式（实拍 vs 非实拍）")
    args = parser.parse_args()
    
    print("=" * 60)
    mode_name = "二分类（实拍 vs 非实拍）" if args.binary else "三分类"
    print(f"  身份证多场景{mode_name} —— 测试评估模块")
    print("=" * 60)
    
    test_pipeline(
        features_dir=args.features_dir,
        model_path=args.model_path,
        confusion_matrix_path=args.confusion_matrix_path,
        binary_mode=args.binary,
    )


if __name__ == "__main__":
    main()