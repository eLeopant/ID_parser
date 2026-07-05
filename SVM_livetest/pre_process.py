"""
pre_process.py —— 身份证多场景三分类预处理模块
======================================================
功能：
  1. 自动检测数据集目录结构（预划分模式 / ID划分模式）
  2. 256×256 无重叠滑动窗口切块（Patching）
  3. 提取 LBP + HOG + HSV 三类手工纹理特征并拼接
  4. 全局标准化（StandardScaler）并保存特征、标签、标准化器

使用方式：
  python pre_process.py --dataset_root dataset/ --output_dir outputs/
  
  或在 main.py 中调用 pre_process_pipeline() 函数

作者：计算机视觉工程
日期：2026-07-05
"""

import os
import sys
import pickle
import argparse
import warnings
from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2
from tqdm import tqdm

from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================================
#  全局配置常量
# ============================================================================

PATCH_SIZE = 256          # 滑动窗口大小（像素）
STRIDE = 256              # 步长（=PATCH_SIZE 表示无重叠）
LAPLACIAN_THRESH = 15     # 拉普拉斯方差阈值，低于该值视为无纹理背景，丢弃

# LBP 参数
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"    # Uniform LBP → 59 bins

# HOG 参数
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# HSV 直方图参数
HSV_H_BINS = 16
HSV_S_BINS = 16

# 类别名称 → 标签映射
CATEGORY_MAP = {
    "real": 0,
    "screen": 1,
    "paper": 2,
    "0_real": 0,
    "1_screen": 1,
    "2_copy": 2,
}


# ============================================================================
#  辅助工具函数
# ============================================================================

def detect_data_mode(dataset_root: str) -> str:
    """
    自动检测数据集目录结构模式。
    
    模式 A（预划分模式）：每类目录内含 train/ 和 test/ 子目录
      例：real/train/, real/test/, screen/train/, screen/test/
    
    模式 B（ID划分模式）：类目录扁平，需按文件名ID前缀划分
      例：0_real/, 1_screen/, 2_copy/
    
    Args:
        dataset_root: 数据集根目录路径
    
    Returns:
        "A" 或 "B"
    """
    subdirs = [d for d in os.listdir(dataset_root) 
               if os.path.isdir(os.path.join(dataset_root, d))]
    
    # 检查是否存在 train/test 子目录结构
    has_train_test = any(
        os.path.exists(os.path.join(dataset_root, d, "train")) and
        os.path.exists(os.path.join(dataset_root, d, "test"))
        for d in subdirs
    )
    
    if has_train_test:
        print("[预处理] 检测到数据集结构：模式 A（预划分 train/test）")
        return "A"
    else:
        print("[预处理] 检测到数据集结构：模式 B（需按 ID 前缀划分）")
        return "B"


def is_textureless_patch(patch_gray: np.ndarray, threshold: float = LAPLACIAN_THRESH) -> bool:
    """
    使用拉普拉斯方差判断切块是否缺乏纹理（纯背景/纯色区域）。
    
    原理：拉普拉斯算子计算图像的二阶导数，平坦区域的方差极低。
    
    Args:
        patch_gray: 灰度切块 (H, W)
        threshold: 方差阈值
    
    Returns:
        True 表示无纹理/应丢弃，False 表示有效纹理
    """
    lap = cv2.Laplacian(patch_gray, cv2.CV_64F)
    variance = lap.var()
    return variance < threshold


def extract_lbp_features(patch_gray: np.ndarray) -> np.ndarray:
    """
    提取 Uniform LBP 局部二值模式特征（59 维归一化直方图）。
    
    Uniform LBP 对光照变化鲁棒，擅长捕捉拍屏摩尔纹、复印件纸张
    粗糙度等微观纹理模式。
    
    Args:
        patch_gray: 灰度切块 (256, 256), dtype uint8
    
    Returns:
        59 维归一化 LBP 直方图向量
    """
    lbp_img = local_binary_pattern(
        patch_gray, P=LBP_P, R=LBP_R, method=LBP_METHOD
    )
    # Uniform LBP 共有 P*(P-1)+3 = 59 种模式
    hist, _ = np.histogram(lbp_img.ravel(), bins=59, range=(0, 59), density=True)
    return hist.astype(np.float32)


def extract_hog_features(patch_gray: np.ndarray) -> np.ndarray:
    """
    提取 HOG（方向梯度直方图）特征。
    
    HOG 捕捉边缘方向分布，对复印文本边缘的碳粉扩散、锯齿感敏感，
    也与拍屏摩尔纹的周期性梯度变化强相关。
    
    Args:
        patch_gray: 灰度切块 (256, 256), dtype uint8
    
    Returns:
        展平的 HOG 特征向量
    """
    hog_feat = hog(
        patch_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        feature_vector=True,
        channel_axis=None,
    )
    return hog_feat.astype(np.float32)


def extract_hsv_features(patch_bgr: np.ndarray) -> np.ndarray:
    """
    提取 HSV 颜色直方图特征（H 16-bin + S 16-bin = 32 维，L1 归一化）。
    
    HSV 色彩空间分离亮度与色彩信息：
    - H（色调）直方图捕捉屏幕蓝光偏色
    - S（饱和度）直方图捕捉复印件低饱和度特征
    
    Args:
        patch_bgr: BGR 切块 (256, 256, 3), dtype uint8
    
    Returns:
        32 维 L1 归一化 HSV 直方图向量
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    # H 通道：16-bin 直方图（范围 0-180）
    h_hist, _ = np.histogram(h_channel.ravel(), bins=HSV_H_BINS, range=(0, 180), density=False)
    # S 通道：16-bin 直方图（范围 0-255）
    s_hist, _ = np.histogram(s_channel.ravel(), bins=HSV_S_BINS, range=(0, 255), density=False)
    
    # L1 归一化
    h_hist = h_hist.astype(np.float32)
    s_hist = s_hist.astype(np.float32)
    
    if h_hist.sum() > 0:
        h_hist /= h_hist.sum()
    if s_hist.sum() > 0:
        s_hist /= s_hist.sum()
    
    combined = np.concatenate([h_hist, s_hist])
    return combined


def extract_patch_features(patch_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    对单个 256×256 BGR 切块提取完整特征向量（LBP + HOG + HSV）。
    
    流程：
      1. BGR → Gray
      2. 提取 LBP（59 维）
      3. 提取 HOG
      4. 提取 HSV（32 维）
      5. 拼接为总特征向量
    
    Args:
        patch_bgr: BGR 切块 (256, 256, 3)
    
    Returns:
        拼接后的一维特征向量；若切块无纹理则返回 None
    """
    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    
    # 过滤无纹理切块
    if is_textureless_patch(patch_gray):
        return None
    
    # 提取三类特征
    lbp_feat = extract_lbp_features(patch_gray)
    hog_feat = extract_hog_features(patch_gray)
    hsv_feat = extract_hsv_features(patch_bgr)
    
    # 横向拼接
    combined = np.concatenate([lbp_feat, hog_feat, hsv_feat])
    return combined


def image_to_patches(image_bgr: np.ndarray) -> List[np.ndarray]:
    """
    将整张 BGR 图像按 256×256 无重叠滑动窗口切块。
    
    丢弃边缘不足 256 的无效区域。
    
    Args:
        image_bgr: 输入 BGR 图像 (H, W, 3)
    
    Returns:
        BGR 切块列表，每个切块 (256, 256, 3)
    """
    h, w = image_bgr.shape[:2]
    patches = []
    
    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = image_bgr[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patches.append(patch)
    
    return patches


# ============================================================================
#  模式 A：预划分数据加载
# ============================================================================

def load_data_mode_a(dataset_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    模式 A：数据集已按 train/test 预划分。
    
    目录结构：
      dataset_root/
        real/
          train/  (实拍训练)
          test/   (实拍测试)
        screen/
          train/  (拍屏训练)
          test/   (拍屏测试)
        paper/
          train/  (复印件训练)
          test/   (复印件测试)
    
    Returns:
        X_train, y_train, X_test, y_test, train_img_paths, test_img_paths
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    train_img_paths, test_img_paths = [], []
    
    categories = [d for d in os.listdir(dataset_root) 
                  if os.path.isdir(os.path.join(dataset_root, d))]
    
    for category in categories:
        cat_path = os.path.join(dataset_root, category)
        label = CATEGORY_MAP.get(category, -1)
        if label == -1:
            print(f"[警告] 未知类别目录: {category}，跳过")
            continue
        
        # 处理训练集
        train_dir = os.path.join(cat_path, "train")
        if os.path.exists(train_dir):
            print(f"[预处理] 加载训练集：{category}/train/ (标签={label})")
            for fname in tqdm(sorted(os.listdir(train_dir)), desc=f"  {category}/train"):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(train_dir, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[警告] 无法读取图像: {img_path}")
                        continue
                    patches = image_to_patches(img)
                    for p_idx, patch in enumerate(patches):
                        feat = extract_patch_features(patch)
                        if feat is not None:
                            X_train_list.append(feat)
                            y_train_list.append(label)
                            train_img_paths.append(f"{img_path}#patch{p_idx}")
        
        # 处理测试集
        test_dir = os.path.join(cat_path, "test")
        if os.path.exists(test_dir):
            print(f"[预处理] 加载测试集：{category}/test/ (标签={label})")
            for fname in tqdm(sorted(os.listdir(test_dir)), desc=f"  {category}/test"):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(test_dir, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[警告] 无法读取图像: {img_path}")
                        continue
                    patches = image_to_patches(img)
                    for p_idx, patch in enumerate(patches):
                        feat = extract_patch_features(patch)
                        if feat is not None:
                            X_test_list.append(feat)
                            y_test_list.append(label)
                            test_img_paths.append(f"{img_path}#patch{p_idx}")
    
    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.empty((0,), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32) if y_train_list else np.empty((0,), dtype=np.int32)
    X_test = np.array(X_test_list, dtype=np.float32) if X_test_list else np.empty((0,), dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32) if y_test_list else np.empty((0,), dtype=np.int32)
    
    return X_train, y_train, X_test, y_test, train_img_paths, test_img_paths


# ============================================================================
#  模式 B：按 ID 前缀划分数据
# ============================================================================

def load_data_mode_b(dataset_root: str, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    模式 B：按身份证 ID 前缀 80/20 划分训练/测试集。
    
    目录结构：
      dataset_root/
        0_real/      (实拍，标签 0)
        1_screen/    (拍屏，标签 1)
        2_copy/      (复印件翻拍，标签 2)
    
    文件命名规则：ID_Device_Condition_Index.jpg
    按 ID（下划线第一段）分组，前 80% ID 归训练集，后 20% 归测试集。
    
    Args:
        dataset_root: 数据集根目录
        train_ratio: 训练集比例（默认 0.8）
    
    Returns:
        X_train, y_train, X_test, y_test, train_img_paths, test_img_paths
    """
    # 收集所有图片路径及标签、ID
    all_images = []
    
    for cat_dir in sorted(os.listdir(dataset_root)):
        cat_path = os.path.join(dataset_root, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        label = CATEGORY_MAP.get(cat_dir, -1)
        if label == -1:
            print(f"[警告] 未知类别目录: {cat_dir}，跳过")
            continue
        
        for fname in sorted(os.listdir(cat_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(cat_path, fname)
                # 提取 ID 前缀（下划线第一段）
                stem = os.path.splitext(fname)[0]
                id_prefix = stem.split('_')[0] if '_' in stem else stem
                all_images.append({
                    'path': img_path,
                    'label': label,
                    'id': id_prefix,
                })
    
    if not all_images:
        print("[错误] 数据集为空！")
        return (np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32),
                [], [])
    
    # 按 ID 分组，去重排序
    unique_ids = sorted(set(img['id'] for img in all_images))
    n_ids = len(unique_ids)
    n_train_ids = max(1, int(n_ids * train_ratio))
    
    train_ids = set(unique_ids[:n_train_ids])
    test_ids = set(unique_ids[n_train_ids:])
    
    print(f"[预处理] 模式 B：共 {n_ids} 个唯一 ID，{len(train_ids)} 个训练，{len(test_ids)} 个测试")
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    train_img_paths, test_img_paths = [], []
    
    for img_info in tqdm(all_images, desc="[预处理] 处理图像"):
        img = cv2.imread(img_info['path'])
        if img is None:
            print(f"[警告] 无法读取图像: {img_info['path']}")
            continue
        
        patches = image_to_patches(img)
        is_train = img_info['id'] in train_ids
        
        for p_idx, patch in enumerate(patches):
            feat = extract_patch_features(patch)
            if feat is not None:
                trace = f"{img_info['path']}#patch{p_idx}"
                if is_train:
                    X_train_list.append(feat)
                    y_train_list.append(img_info['label'])
                    train_img_paths.append(trace)
                else:
                    X_test_list.append(feat)
                    y_test_list.append(img_info['label'])
                    test_img_paths.append(trace)
    
    X_train = np.array(X_train_list, dtype=np.float32) if X_train_list else np.empty((0,), dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32) if y_train_list else np.empty((0,), dtype=np.int32)
    X_test = np.array(X_test_list, dtype=np.float32) if X_test_list else np.empty((0,), dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int32) if y_test_list else np.empty((0,), dtype=np.int32)
    
    return X_train, y_train, X_test, y_test, train_img_paths, test_img_paths


# ============================================================================
#  主预处理管道
# ============================================================================

def pre_process_pipeline(
    dataset_root: str = "dataset/",
    output_dir: str = "outputs/",
    skip_if_exists: bool = False,
    binary_mode: bool = False,
) -> Dict[str, str]:
    """
    完整预处理管道：加载数据 → 切块 → 特征提取 → 标准化 → 保存。
    
    Args:
        dataset_root: 数据集根目录路径
        output_dir: 输出目录路径
        skip_if_exists: 若已存在特征文件，是否跳过预处理
        binary_mode: True=二分类模式（实拍=0 vs 非实拍=1），False=三分类模式
    
    Returns:
        包含已保存文件路径的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件路径
    paths = {
        "X_train": os.path.join(output_dir, "X_train.pkl"),
        "y_train": os.path.join(output_dir, "y_train.pkl"),
        "X_test": os.path.join(output_dir, "X_test.pkl"),
        "y_test": os.path.join(output_dir, "y_test.pkl"),
        "scaler": os.path.join(output_dir, "scaler.pkl"),
    }
    
    # 断点复用检查
    if skip_if_exists and all(os.path.exists(p) for p in paths.values()):
        print("[预处理] 所有特征文件已存在，跳过预处理（skip_if_exists=True）")
        return paths
    
    # Step 1: 检测数据模式并加载
    mode = detect_data_mode(dataset_root)
    
    if mode == "A":
        X_train, y_train, X_test, y_test, train_paths, test_paths = load_data_mode_a(dataset_root)
    else:
        X_train, y_train, X_test, y_test, train_paths, test_paths = load_data_mode_b(dataset_root)
    
    # 数据校验
    if X_train.size == 0 and X_test.size == 0:
        raise RuntimeError("[错误] 训练集和测试集均为空！请检查数据集路径和图像格式。")
    
    print(f"\n[预处理] ========== 数据统计 ==========")
    print(f"  训练集切块数: {len(X_train)}")
    print(f"  测试集切块数: {len(X_test)}")
    print(f"  特征维度:    {X_train.shape[1] if X_train.size > 0 else X_test.shape[1]}")
    for cls_id, cls_name in [(0, "实拍(real)"), (1, "拍屏(screen)"), (2, "复印件(paper)")]:
        train_cnt = np.sum(y_train == cls_id) if y_train.size > 0 else 0
        test_cnt = np.sum(y_test == cls_id) if y_test.size > 0 else 0
        print(f"  {cls_name}: 训练={train_cnt}, 测试={test_cnt}")
    
    # Step 1.5: 二分类模式——将拍屏(1)和复印件(2)合并为"非实拍"(1)
    if binary_mode:
        print(f"\n[预处理] ⚡ 二分类模式：拍屏(1) + 复印件(2) → 非实拍(1)")
        if y_train.size > 0:
            y_train = np.where(y_train == 2, 1, y_train)  # 标签2→1
        if y_test.size > 0:
            y_test = np.where(y_test == 2, 1, y_test)
        for cls_id, cls_name in [(0, "实拍"), (1, "非实拍")]:
            train_cnt = np.sum(y_train == cls_id) if y_train.size > 0 else 0
            test_cnt = np.sum(y_test == cls_id) if y_test.size > 0 else 0
            print(f"  {cls_name}(标签={cls_id}): 训练={train_cnt}, 测试={test_cnt}")
    
    # Step 2: 标准化（仅在训练集上拟合）
    scaler = StandardScaler()
    if X_train.size > 0:
        print("\n[预处理] 在训练集上拟合 StandardScaler ...")
        X_train = scaler.fit_transform(X_train).astype(np.float32)
    if X_test.size > 0 and X_train.size > 0:
        print("[预处理] 对测试集应用标准化 ...")
        X_test = scaler.transform(X_test).astype(np.float32)
    
    # Step 3: 保存
    print("\n[预处理] 保存特征文件...")
    with open(paths["X_train"], "wb") as f:
        pickle.dump(X_train, f)
    with open(paths["y_train"], "wb") as f:
        pickle.dump(y_train, f)
    with open(paths["X_test"], "wb") as f:
        pickle.dump(X_test, f)
    with open(paths["y_test"], "wb") as f:
        pickle.dump(y_test, f)
    with open(paths["scaler"], "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"[预处理] ✓ 文件已保存至 {output_dir}/")
    for k, v in paths.items():
        print(f"  {v}")
    
    return paths


# ============================================================================
#  CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="身份证图像预处理（切块+特征提取+标准化）")
    parser.add_argument("--dataset_root", type=str, default="dataset/", help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="输出目录")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="若已存在则跳过")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  身份证多场景三分类 —— 预处理模块")
    print("=" * 60)
    
    pre_process_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        skip_if_exists=args.skip_if_exists,
    )
    
    print("\n[预处理] 完成！")


if __name__ == "__main__":
    main()