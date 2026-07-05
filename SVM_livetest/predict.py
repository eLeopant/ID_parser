"""
predict.py —— 活体检测单张推理脚本
=====================================
功能：
  1. 加载已训练的手写 SVM 模型（OvO 三分类）与标准化器
  2. 读取 task/task.jpg，256×256 无重叠切块
  3. 对每个切块提取 LBP+HOG+HSV 特征，标准化后逐块预测
  4. 多数投票：最多切块预测为类别 0（实拍）→ is_real=1，否则 is_real=0
  5. 将结果写入 task/is_real.yaml

用法：
  python predict.py

  或通过命令行参数指定输入图片和模型路径：
  python predict.py --image task/task.jpg --model outputs/best_model.pkl

依赖：
  - 已执行过 main.py 完成训练，生成 outputs/best_model.pkl 和 outputs/scaler.pkl
  - pre_process.py（同目录，提供特征提取函数）

作者：计算机视觉工程
日期：2026-07-05
"""

import os
import sys
import pickle
import argparse
import warnings
from collections import Counter
from typing import Tuple, List

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2hsv

warnings.filterwarnings("ignore")

# ============================================================================
#  配置常量
# ============================================================================

PATCH_SIZE = 256            # 切块大小
STRIDE = 256                # 步长（等于 PATCH_SIZE → 无重叠）
LAPLACIAN_THRESH = 15       # 无纹理切块过滤阈值（拉普拉斯方差）

# LBP 参数
LBP_P = 8
LBP_R = 1
LBP_N_BINS = 59             # Uniform LBP (P=8 → 10 种 uniform + 1 种 non-uniform = 59 bins)

# HOG 参数
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# HSV 直方图参数
HSV_H_BINS = 16
HSV_S_BINS = 16


# ============================================================================
#  特征提取函数（独立实现，不依赖 pre_process.py，保证单文件可运行）
# ============================================================================

def _patch_is_textureless(patch: np.ndarray, laplacian_thresh: float = LAPLACIAN_THRESH) -> bool:
    """
    通过拉普拉斯方差判断切块是否为纯背景 / 无纹理区域。
    
    Args:
        patch: RGB 图像切块 (H, W, 3)
        laplacian_thresh: 方差阈值，低于此值视为无纹理
    
    Returns:
        True 表示无纹理（应丢弃），False 表示有纹理（保留）
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var() < laplacian_thresh


def extract_lbp_features(patch: np.ndarray, n_bins: int = LBP_N_BINS) -> np.ndarray:
    """
    对 RGB 切块提取 Uniform LBP 归一化直方图特征。
    
    Args:
        patch: RGB 图像切块 (H, W, 3)
        n_bins: 直方图 bins 数（Uniform LBP P=8 对应 59 bins）
    
    Returns:
        1D 特征向量，shape (n_bins,)，L1 归一化
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum  # L1 归一化
    return hist


def extract_hog_features(patch: np.ndarray) -> np.ndarray:
    """
    对 RGB 切块提取 HOG 特征并展平。
    
    Args:
        patch: RGB 图像切块 (H, W, 3)
    
    Returns:
        1D HOG 特征向量
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
    )
    return features


def extract_hsv_features(patch: np.ndarray) -> np.ndarray:
    """
    对 RGB 切块提取 HSV 颜色直方图特征（H 16 bins + S 16 bins = 32 维）。
    
    Args:
        patch: RGB 图像切块 (H, W, 3)，dtype 为 uint8 [0,255]
    
    Returns:
        1D 特征向量，shape (32,)，L1 归一化
    """
    hsv = rgb2hsv(patch)  # scikit-image: 返回 float [0,1]
    # H 通道: [0, 1) → 16 bins
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    h_hist, _ = np.histogram(h_channel, bins=HSV_H_BINS, range=(0, 1), density=False)
    s_hist, _ = np.histogram(s_channel, bins=HSV_S_BINS, range=(0, 1), density=False)
    
    combined = np.concatenate([h_hist, s_hist]).astype(np.float64)
    combined_sum = combined.sum()
    if combined_sum > 0:
        combined /= combined_sum  # L1 归一化
    return combined


def extract_features_from_patch(patch: np.ndarray) -> np.ndarray:
    """
    对单个 256×256 切块提取完整特征向量（LBP + HOG + HSV 拼接）。
    
    Args:
        patch: RGB 图像切块 (256, 256, 3)
    
    Returns:
        1D 特征向量，LBP(59) + HOG(?) + HSV(32)
    """
    lbp_feat = extract_lbp_features(patch)       # (59,)
    hog_feat = extract_hog_features(patch)        # depends on patch/cell/block
    hsv_feat = extract_hsv_features(patch)        # (32,)
    return np.concatenate([lbp_feat, hog_feat, hsv_feat])


def sliding_window_patches(
    image: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    laplacian_thresh: float = LAPLACIAN_THRESH,
) -> List[np.ndarray]:
    """
    对整张图片执行无重叠滑动窗口切块，过滤无纹理区域。
    
    Args:
        image: RGB 图像 (H, W, 3)
        patch_size: 切块边长
        stride: 滑动步长
        laplacian_thresh: 无纹理过滤阈值
    
    Returns:
        有效切块列表，每个切块 shape (256, 256, 3)
    """
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            if not _patch_is_textureless(patch, laplacian_thresh):
                patches.append(patch)
    
    return patches


# ============================================================================
#  模型加载
# ============================================================================

def load_model_and_scaler(
    model_path: str = "outputs/best_model.pkl",
    scaler_path: str = "outputs/scaler.pkl",
):
    """
    加载手写 SVM 模型（OvO 三分类）与标准化器。
    
    Args:
        model_path: 模型 pkl 文件路径
        scaler_path: 标准化器 pkl 文件路径
    
    Returns:
        (model, scaler)
    """
    from train import OvOSVMClassifier
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}。请先运行 main.py 完成训练。")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}。请先运行 main.py 完成预处理。")
    
    model = OvOSVMClassifier.load(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print(f"[加载] 模型: {model_path} ({model.n_classes_} 类, {len(model.classifiers)} 个二分类器)")
    print(f"[加载] 标准化器: {scaler_path}")
    
    return model, scaler


# ============================================================================
#  单张图片推理
# ============================================================================

def predict_single_image(
    image_path: str,
    model,
    scaler,
) -> Tuple[int, dict]:
    """
    对单张图片执行活体检测推理。
    
    流程：
      1. 读取图片 → 256×256 无重叠切块
      2. 对每个切块提取 LBP+HOG+HSV 特征
      3. 标准化特征
      4. 逐块三分类预测（0=实拍, 1=拍屏, 2=复印件）
      5. 多数投票决定最终类别
    
    Args:
        image_path: 输入图片路径
        model: OvOSVMClassifier 实例
        scaler: StandardScaler 实例
    
    Returns:
        (final_class, stats_dict)
        final_class: 0=实拍, 1=拍屏, 2=复印件
        stats_dict: {
            "total_patches": 总有效切块数,
            "votes": {0: 票数, 1: 票数, 2: 票数},
            "real_ratio": 实拍票数占比,
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    # Step 1: 读取图片并切块
    print(f"\n[推理] 读取图片: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  图片尺寸: {image_rgb.shape[1]}×{image_rgb.shape[0]}")
    
    patches = sliding_window_patches(image_rgb)
    n_total = len(patches)
    print(f"  有效切块数: {n_total} (已过滤无纹理区域)")
    
    if n_total == 0:
        print("  [警告] 未提取到任何有效切块，判定为非实拍")
        return 1, {"total_patches": 0, "votes": {0: 0, 1: 0, 2: 0}, "real_ratio": 0.0}
    
    # Step 2: 逐块提取特征
    print(f"  提取特征中... (LBP {LBP_N_BINS}维 + HOG + HSV 32维)")
    features_list = []
    for patch in patches:
        feat = extract_features_from_patch(patch)
        features_list.append(feat)
    
    X = np.array(features_list, dtype=np.float64)
    print(f"  特征矩阵: {X.shape}")
    
    # Step 3: 标准化
    X_scaled = scaler.transform(X)
    
    # Step 4: 逐块预测
    print(f"  执行逐块预测...")
    predictions = model.predict(X_scaled)  # shape (n_patches,)
    print(f"  预测完成，共 {len(predictions)} 个切块")
    
    # Step 5: 多数投票
    vote_counter = Counter(predictions)
    votes = {0: vote_counter.get(0, 0), 1: vote_counter.get(1, 0), 2: vote_counter.get(2, 0)}
    final_class = max(votes, key=votes.get)  # 得票最多的类别
    real_ratio = votes[0] / n_total if n_total > 0 else 0.0
    
    stats = {
        "total_patches": n_total,
        "votes": votes,
        "real_ratio": real_ratio,
    }
    
    return final_class, stats


# ============================================================================
#  结果写入 YAML
# ============================================================================

def write_result_yaml(
    output_path: str,
    is_real: int,
    stats: dict = None,
):
    """
    将活体检测结果写入 YAML 文件。
    
    格式:
        is_real: 1   (实拍)
        is_real: 0   (非实拍)
    
    Args:
        output_path: YAML 输出路径
        is_real: 1=实拍, 0=非实拍
        stats: 可选的详细统计信息（会以注释形式写入）
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    lines = []
    lines.append(f"# 活体检测结果")
    lines.append(f"# 生成方式: predict.py 逐块预测 + 多数投票")
    if stats:
        total = stats["total_patches"]
        votes = stats["votes"]
        lines.append(f"# 总有效切块数: {total}")
        lines.append(f"# 投票分布: 实拍={votes[0]}, 拍屏={votes[1]}, 复印件={votes[2]}")
        if total > 0:
            lines.append(f"# 实拍占比: {stats['real_ratio']:.2%}")
    lines.append(f"is_real: {is_real}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"\n[输出] 结果已写入: {output_path}")
    print(f"  is_real: {is_real}")


# ============================================================================
#  主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="活体检测 —— 单张身份证图片推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python predict.py
  python predict.py --image task/task.jpg --model outputs/best_model.pkl
        """,
    )
    parser.add_argument("--image", type=str, default="task/task.jpg", help="输入图片路径")
    parser.add_argument("--model", type=str, default="outputs/best_model.pkl", help="模型路径")
    parser.add_argument("--scaler", type=str, default="outputs/scaler.pkl", help="标准化器路径")
    parser.add_argument("--output", type=str, default="task/is_real.yaml", help="输出 YAML 路径")
    parser.add_argument("--patch_size", type=int, default=256, help="切块大小")
    parser.add_argument("--laplacian_thresh", type=float, default=15.0, help="无纹理切块过滤阈值")
    args = parser.parse_args()
    
    # 覆写全局常量
    global PATCH_SIZE, STRIDE, LAPLACIAN_THRESH
    PATCH_SIZE = args.patch_size
    STRIDE = args.patch_size  # 无重叠
    LAPLACIAN_THRESH = args.laplacian_thresh
    
    print("=" * 60)
    print("  身份证活体检测 —— 单张推理")
    print("  模型: 手写 RBF 核 SVM (OvO 三分类)")
    print("=" * 60)
    print(f"  输入图片:   {args.image}")
    print(f"  模型文件:   {args.model}")
    print(f"  标准化器:   {args.scaler}")
    print(f"  输出 YAML:  {args.output}")
    print()
    
    # Step 1: 加载模型与标准化器
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    
    # Step 2: 推理
    final_class, stats = predict_single_image(args.image, model, scaler)
    
    # Step 3: 输出结果
    class_names = {0: "实拍 (Real)", 1: "拍屏 (Screen)", 2: "复印件 (Paper)"}
    print(f"\n{'=' * 60}")
    print(f"  推理结果")
    print(f"{'=' * 60}")
    print(f"  预测类别: {final_class} ({class_names[final_class]})")
    print(f"  投票分布: 实拍={stats['votes'][0]}, 拍屏={stats['votes'][1]}, 复印件={stats['votes'][2]}")
    if stats["total_patches"] > 0:
        print(f"  实拍占比: {stats['real_ratio']:.2%} ({stats['votes'][0]}/{stats['total_patches']})")
    
    # 判定 is_real：预测为类别 0（实拍）→ 1，否则 → 0
    is_real = 1 if final_class == 0 else 0
    print(f"  is_real: {is_real} {'(实拍 ✓)' if is_real == 1 else '(非实拍 ✗)'}")
    
    # Step 4: 写入 YAML
    write_result_yaml(args.output, is_real, stats)
    
    print(f"\n{'=' * 60}")
    print(f"  推理完成")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()