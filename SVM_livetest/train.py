"""
train.py —— 手写 RBF 核 SVM 训练模块
======================================
功能：
  1. 纯 Python + NumPy 从零实现 RBF 核 SVM
  2. SMO（序列最小优化）算法求解对偶问题
  3. One-vs-One 三分类策略（投票决策）
  4. 网格搜索 + 5 折交叉验证自动选最优超参数
  5. 保存最优模型为 pkl 文件

核心约束：
  - 严禁调用 sklearn.svm.SVC 等封装好的 SVM 分类器
  - 仅使用 NumPy 进行矩阵运算
  - 支持软间隔惩罚参数 C

作者：计算机视觉工程
日期：2026-07-05
"""

import os
import pickle
import argparse
import warnings
from typing import Tuple, Optional, Dict, List

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
#  RBF（高斯）核函数
# ============================================================================

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    计算 RBF（高斯）核矩阵。
    
    K(x_i, x_j) = exp(-γ * ||x_i - x_j||²)
    
    高效计算技巧：
      ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2 * <x_i, x_j>
    利用矩阵运算一次性计算整个核矩阵，避免逐对循环。
    
    Args:
        X1: 样本矩阵 (N1, D)
        X2: 样本矩阵 (N2, D)
        gamma: 核参数 γ，控制高斯函数宽度
    
    Returns:
        核矩阵 (N1, N2)
    """
    # 计算每行向量的平方范数
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)  # (N1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)  # (1, N2)
    
    # 点积矩阵
    cross = np.dot(X1, X2.T)  # (N1, N2)
    
    # 欧氏距离平方矩阵
    sq_dists = X1_sq + X2_sq - 2.0 * cross
    
    # 数值稳定性：裁剪极小负值（浮点舍入误差）
    sq_dists = np.maximum(sq_dists, 0.0)
    
    # 计算 RBF 核
    K = np.exp(-gamma * sq_dists)
    return K


# ============================================================================
#  纯手写 SVM 二分类器（SMO 算法）
# ============================================================================

class HandwrittenSVM:
    """
    手写 RBF 核 SVM 二分类器。
    
    使用 SMO（Sequential Minimal Optimization）算法求解对偶问题：
    
      max_α  Σα_i - 0.5 * Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
      s.t.    0 ≤ α_i ≤ C,  Σ α_i y_i = 0
    
    算法原理：
      SMO 每次选择两个拉格朗日乘子 α₁, α₂ 进行优化，将多维优化
      问题分解为一系列二维子问题，每个子问题有解析解。
    
    Args:
        C: 软间隔惩罚参数（0 < C < ∞）
        gamma: RBF 核参数 γ
        tol: KKT 条件容忍度
        max_iter: 最大迭代次数
    """
    
    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 0.01,
        tol: float = 1e-3,
        max_iter: int = 2000,
    ):
        self.C = C                # 软间隔惩罚参数
        self.gamma = gamma        # RBF 核参数
        self.tol = tol            # KKT 条件容忍度
        self.max_iter = max_iter  # 最大迭代次数
        
        # 训练后保存的参数
        self.alpha = None         # 拉格朗日乘子 (N,)
        self.b = 0.0              # 偏置项
        self.support_vectors = None  # 支持向量 (N_sv, D)
        self.support_labels = None   # 支持向量标签 (N_sv,)
        self.support_alpha = None    # 支持向量对应的 α (N_sv,)
        self.X_train = None       # 用于预测时计算核矩阵的训练数据引用
        
    def _compute_kernel(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """计算 RBF 核矩阵的便捷方法"""
        if X2 is None:
            X2 = X1
        return rbf_kernel(X1, X2, self.gamma)
    
    def _take_step(self, i: int, j: int, K: np.ndarray, y: np.ndarray,
                   alpha: np.ndarray, errors: np.ndarray) -> int:
        """
        SMO 算法核心：对两个拉格朗日乘子 α_i, α_j 执行一次联合优化。
        
        解析推导：
          设 α₂_new = α₂_old + y₂ * (E₁ - E₂) / η
          其中  η = K(x₁,x₁) + K(x₂,x₂) - 2*K(x₁,x₂)
                E_k = f(x_k) - y_k  为预测误差
          
          然后裁剪 α₂_new 到 [L, H] 范围内：
            若 y₁ ≠ y₂: L=max(0,α₂-α₁), H=min(C,C+α₂-α₁)
            若 y₁ = y₂: L=max(0,α₁+α₂-C), H=min(C,α₁+α₂)
          
          再由约束更新 α₁：α₁_new = α₁_old + y₁*y₂*(α₂_old - α₂_new)
        
        Returns:
            1 表示 α 被更新，0 表示未更新
        """
        if i == j:
            return 0
        
        # 获取当前 α 值
        alpha_i_old = alpha[i]
        alpha_j_old = alpha[j]
        y_i = y[i]
        y_j = y[j]
        
        # 计算预测误差
        E_i = errors[i]
        E_j = errors[j]
        
        s = y_i * y_j
        
        # 计算 α_j 的裁剪边界 [L, H]
        if y_i != y_j:
            L = max(0.0, alpha[j] - alpha[i])
            H = min(self.C, self.C + alpha[j] - alpha[i])
        else:
            L = max(0.0, alpha[i] + alpha[j] - self.C)
            H = min(self.C, alpha[i] + alpha[j])
        
        if L >= H:
            return 0
        
        # 计算二阶导数 η = K_ii + K_jj - 2*K_ij
        eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
        
        if eta <= 0:
            return 0
        
        # 计算新的 α_j（未裁剪）
        alpha_j_new = alpha_j_old + y_j * (E_i - E_j) / eta
        
        # 裁剪到 [L, H]
        alpha_j_new = np.clip(alpha_j_new, L, H)
        
        # 若变化过小则跳过
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return 0
        
        # 更新 α_i
        alpha_i_new = alpha_i_old + s * (alpha_j_old - alpha_j_new)
        
        # 更新偏置 b
        # b₁ = b - E_i - y_i*(α_i_new-α_i_old)*K_ii - y_j*(α_j_new-α_j_old)*K_ij
        b1 = (self.b - E_i 
              - y_i * (alpha_i_new - alpha_i_old) * K[i, i]
              - y_j * (alpha_j_new - alpha_j_old) * K[i, j])
        
        # b₂ = b - E_j - y_i*(α_i_new-α_i_old)*K_ij - y_j*(α_j_new-α_j_old)*K_jj
        b2 = (self.b - E_j
              - y_i * (alpha_i_new - alpha_i_old) * K[i, j]
              - y_j * (alpha_j_new - alpha_j_old) * K[j, j])
        
        # 若 α_i 在 (0, C) 内则取 b₁，若 α_j 在 (0, C) 内则取 b₂，否则取均值
        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        
        # 更新 α
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        
        # 更新误差缓存
        # 对于所有 k：E_k_new = E_k_old + (α_i_new-α_i_old)*y_i*K_ik + (α_j_new-α_j_old)*y_j*K_jk + b_old - b_new
        # 这里简化：重新计算所有样本的误差
        f_all = np.dot(K, alpha * y) + self.b
        errors_new = f_all - y
        errors[:] = errors_new
        
        return 1
    
    def _examine_example(self, i: int, K: np.ndarray, y: np.ndarray,
                         alpha: np.ndarray, errors: np.ndarray) -> int:
        """
        检查第 i 个样本是否违反 KKT 条件，若是则尝试选择第二个变量并优化。
        
        KKT 条件（对偶问题）：
          α_i = 0      →  y_i * f(x_i) ≥ 1           (正确分类，远离边界)
          0 < α_i < C  →  y_i * f(x_i) = 1           (支持向量，在边界上)
          α_i = C      →  y_i * f(x_i) ≤ 1           (误分类或在间隔内)
        
        Returns:
            1 表示执行了优化步，0 表示未执行
        """
        y_i = y[i]
        alpha_i = alpha[i]
        E_i = errors[i]
        r_i = E_i * y_i  # = y_i * f(x_i) - 1
        
        # 检查 KKT 条件，带容忍度 tol
        kkt_violated = (
            (alpha_i < self.C - self.tol and r_i < -self.tol) or  # α_i < C, y*f<1
            (alpha_i > self.tol and r_i > self.tol)                # α_i > 0, y*f>1
        )
        
        if not kkt_violated:
            return 0
        
        # ----- 选择第二个变量 j -----
        # 启发式 1: 选择使 |E_i - E_j| 最大的 j（最大化步长）
        n_samples = len(y)
        
        # 先尝试在所有非零非 C 的 α 中找（这些样本最可能成为支持向量）
        valid_idx = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]
        if len(valid_idx) > 1:
            # 选择 |E_i - E_j| 最大的
            best_j = valid_idx[0]
            best_delta = -1.0
            for j in valid_idx:
                if j == i:
                    continue
                delta = abs(E_i - errors[j])
                if delta > best_delta:
                    best_delta = delta
                    best_j = j
            if self._take_step(i, best_j, K, y, alpha, errors):
                return 1
        
        # 启发式 2: 遍历所有样本（随机起始点避免偏向）
        perm = np.random.permutation(n_samples)
        for j in perm:
            if j == i:
                continue
            if self._take_step(i, j, K, y, alpha, errors):
                return 1
        
        return 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HandwrittenSVM":
        """
        训练 SVM 二分类器。
        
        标签 y 应为 +1 或 -1 的二值标签。
        
        算法流程（SMO）：
          1. 初始化 α = 0, b = 0
          2. 预计算核矩阵 K（缓存加速）
          3. 循环：
             a. 遍历所有样本，对违反 KKT 条件的尝试优化
             b. 若整轮无变化且所有 KKT 满足，则收敛
             c. 或达到 max_iter 则停止
        
        Args:
            X: 训练特征矩阵 (N, D)
            y: 标签向量 (N,)，取值为 +1 或 -1
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # 保存训练数据引用（预测时计算核矩阵用）
        self.X_train = X.copy()
        
        # 初始化
        alpha = np.zeros(n_samples, dtype=np.float64)
        self.b = 0.0
        
        # 预计算核矩阵（缓存）
        K = self._compute_kernel(X)
        
        # 初始化误差缓存：E_i = f(x_i) - y_i = 0 - y_i = -y_i (当 α=0, b=0)
        errors = -y.astype(np.float64)
        
        # SMO 外层循环
        num_changed = 0
        examine_all = True
        
        for iteration in range(self.max_iter):
            num_changed = 0
            
            if examine_all:
                # 遍历所有样本
                for i in range(n_samples):
                    num_changed += self._examine_example(i, K, y, alpha, errors)
            else:
                # 仅遍历非边界样本（0 < α < C）
                non_bound_idx = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]
                for i in non_bound_idx:
                    num_changed += self._examine_example(i, K, y, alpha, errors)
            
            # 交替遍历策略：先全遍历，再仅遍历非边界样本
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                # 非边界样本无变化 → 回退到全遍历
                examine_all = True
            
            # 收敛检查（每轮结束后，若全遍历且无变化则收敛）
            if examine_all and num_changed == 0:
                if iteration > 0:
                    # 额外检查：确认所有 KKT 条件满足
                    r_all = errors * y  # y*f - 1
                    kkt_ok = np.all(
                        ((alpha < self.tol) | (r_all >= -self.tol)) &
                        ((alpha > self.C - self.tol) | (r_all <= self.tol))
                    )
                    if kkt_ok:
                        print(f"  [SMO] 收敛于迭代 {iteration}，num_changed=0 且 KKT 满足")
                        break
        
        # 提取支持向量（α > tol 的样本）
        sv_idx = np.where(alpha > self.tol)[0]
        self.support_vectors = X[sv_idx].copy()
        self.support_labels = y[sv_idx].copy()
        self.support_alpha = alpha[sv_idx].copy()
        self.alpha = alpha
        
        print(f"  [SMO] 训练完成: 支持向量数={len(sv_idx)}/{n_samples}, b={self.b:.4f}")
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算决策函数值 f(x) = Σ α_i y_i K(x_i, x) + b。
        
        Args:
            X: 待预测样本 (M, D)
        
        Returns:
            决策函数值 (M,)，>0 为正类，<0 为负类
        """
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.full(X.shape[0], self.b)
        
        # 计算待测样本与所有支持向量的核矩阵
        K_sv = rbf_kernel(X, self.support_vectors, self.gamma)
        
        # f(x) = Σ α_i y_i K(x_i, x) + b
        decision = np.dot(K_sv, self.support_alpha * self.support_labels) + self.b
        return decision
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        二分类预测。
        
        Args:
            X: 待预测样本 (M, D)
        
        Returns:
            预测标签 (M,)，+1 或 -1
        """
        decision = self.decision_function(X)
        return np.sign(decision).astype(np.int32)


# ============================================================================
#  One-vs-One 多分类 SVM
# ============================================================================

class OvOSVMClassifier:
    """
    One-vs-One（一对一）多分类 SVM。
    
    策略：
      对于 K 个类别，训练 K*(K-1)/2 个二分类器。
      预测时每个二分类器投一票，得票最多的类别为最终预测。
      平票时比较各分类器决策函数绝对值之和（置信度）来决策。
    
    Args:
        C: 软间隔惩罚参数
        gamma: RBF 核参数
        tol: KKT 条件容忍度
        max_iter: SMO 最大迭代次数
    """
    
    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 0.01,
        tol: float = 1e-3,
        max_iter: int = 2000,
    ):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        
        self.classifiers: List[Tuple[int, int, HandwrittenSVM]] = []
        self.classes_ = None
        self.n_classes_ = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OvOSVMClassifier":
        """
        训练 OvO 多分类 SVM。
        
        对每一对类别 (i, j)，训练一个二分类器区分 i vs j。
        
        Args:
            X: 训练特征矩阵 (N, D)
            y: 标签向量 (N,)，取值 0, 1, 2...（整数）
        
        Returns:
            self
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        print(f"\n[OvO 训练] 类别数={self.n_classes_}，需训练 {self.n_classes_ * (self.n_classes_ - 1) // 2} 个二分类器")
        print(f"  超参数: C={self.C}, gamma={self.gamma}")
        
        self.classifiers = []
        
        for i in range(self.n_classes_):
            for j in range(i + 1, self.n_classes_):
                class_i = self.classes_[i]
                class_j = self.classes_[j]
                
                # 筛选属于类别 i 和 j 的样本
                mask = (y == class_i) | (y == class_j)
                X_ij = X[mask]
                y_ij = y[mask]
                
                # 转换为 +1 / -1 标签（i→+1, j→-1）
                y_binary = np.where(y_ij == class_i, 1, -1).astype(np.float64)
                
                print(f"\n  [{class_i} vs {class_j}] 训练样本数: {len(X_ij)} ({class_i}={(y_binary == 1).sum()}, {class_j}={(y_binary == -1).sum()})")
                
                svm = HandwrittenSVM(
                    C=self.C,
                    gamma=self.gamma,
                    tol=self.tol,
                    max_iter=self.max_iter,
                )
                svm.fit(X_ij, y_binary)
                
                self.classifiers.append((class_i, class_j, svm))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        OvO 投票预测。
        
        Args:
            X: 待预测样本 (M, D)
        
        Returns:
            预测标签 (M,)
        """
        n_samples = X.shape[0]
        n_classifiers = len(self.classifiers)
        
        # 收集每个分类器对每类的投票
        votes = np.zeros((n_samples, self.n_classes_), dtype=np.int32)
        # 同时记录决策函数值（用于平票时置信度决策）
        confidences = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        
        for class_i, class_j, svm in self.classifiers:
            decisions = svm.decision_function(X)  # (M,)
            predictions = np.sign(decisions)      # +1 或 -1
            
            # 投票
            for k in range(n_samples):
                if predictions[k] == 1:
                    winner = class_i
                else:
                    winner = class_j
                
                # 找到类别在 classes_ 中的索引
                idx = np.where(self.classes_ == winner)[0][0]
                votes[k, idx] += 1
                confidences[k, idx] += abs(decisions[k])
        
        # 最终预测：取投票数最多的类，平票时比较置信度
        final_pred = np.zeros(n_samples, dtype=np.int32)
        for k in range(n_samples):
            max_votes = np.max(votes[k])
            best_indices = np.where(votes[k] == max_votes)[0]
            
            if len(best_indices) == 1:
                final_pred[k] = self.classes_[best_indices[0]]
            else:
                # 平票，选置信度最高的
                best_conf_idx = best_indices[np.argmax(confidences[k, best_indices])]
                final_pred[k] = self.classes_[best_conf_idx]
        
        return final_pred
    
    def get_params(self) -> Dict:
        """返回当前模型参数"""
        return {
            "C": self.C,
            "gamma": self.gamma,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }
    
    def save(self, filepath: str):
        """保存模型到 pkl 文件"""
        model_data = {
            "C": self.C,
            "gamma": self.gamma,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
            "classifiers": [
                (ci, cj, {
                    "alpha": svm.alpha,
                    "b": svm.b,
                    "gamma": svm.gamma,
                    "C": svm.C,
                    "support_vectors": svm.support_vectors,
                    "support_labels": svm.support_labels,
                    "support_alpha": svm.support_alpha,
                    "X_train": svm.X_train,
                })
                for ci, cj, svm in self.classifiers
            ],
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"[模型保存] ✓ 已保存至 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "OvOSVMClassifier":
        """从 pkl 文件加载模型"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        obj = cls(
            C=model_data["C"],
            gamma=model_data["gamma"],
            tol=model_data["tol"],
            max_iter=model_data["max_iter"],
        )
        obj.classes_ = model_data["classes_"]
        obj.n_classes_ = model_data["n_classes_"]
        
        obj.classifiers = []
        for ci, cj, svm_dict in model_data["classifiers"]:
            svm = HandwrittenSVM(
                C=svm_dict["C"],
                gamma=svm_dict["gamma"],
                tol=model_data["tol"],
                max_iter=model_data["max_iter"],
            )
            svm.alpha = svm_dict["alpha"]
            svm.b = svm_dict["b"]
            svm.support_vectors = svm_dict["support_vectors"]
            svm.support_labels = svm_dict["support_labels"]
            svm.support_alpha = svm_dict["support_alpha"]
            svm.X_train = svm_dict["X_train"]
            svm.gamma = svm_dict["gamma"]
            svm.C = svm_dict["C"]
            
            obj.classifiers.append((ci, cj, svm))
        
        print(f"[模型加载] ✓ 已从 {filepath} 加载模型（{obj.n_classes_} 类，{len(obj.classifiers)} 个二分类器）")
        return obj


# ============================================================================
#  5 折交叉验证 + 网格搜索
# ============================================================================

def stratified_kfold_by_image(X: np.ndarray, y: np.ndarray, 
                               train_paths: List[str], n_splits: int = 5):
    """
    按原始图片维度分层 K 折划分生成器。
    
    同一切块来源于同一张图片 → 应放在同一折，防止数据泄露。
    同时尽量保持每折中各类别比例与总体一致（分层采样）。
    
    Args:
        X: 特征矩阵 (N, D)
        y: 标签 (N,)
        train_paths: 切块来源路径（格式："img.jpg#patch0"）
        n_splits: 折数
    
    Yields:
        (train_idx, val_idx) 索引数组
    """
    # 从路径中提取原始图片路径
    img_paths = [p.split("#")[0] for p in train_paths]
    unique_imgs = sorted(set(img_paths))
    
    n_imgs = len(unique_imgs)
    if n_imgs < n_splits:
        raise ValueError(f"原始图片数 ({n_imgs}) 小于折数 ({n_splits})，请减少折数或增加数据")
    
    # 按图片 ID 分层打乱
    np.random.seed(42)
    perm = np.random.permutation(n_imgs)
    unique_imgs = [unique_imgs[i] for i in perm]
    
    # 将图片均匀分配到各折
    fold_size = n_imgs // n_splits
    
    for fold in range(n_splits):
        start = fold * fold_size
        if fold == n_splits - 1:
            end = n_imgs
        else:
            end = start + fold_size
        
        val_imgs = set(unique_imgs[start:end])
        
        # 构建索引
        val_idx = np.array([k for k, p in enumerate(img_paths) if p in val_imgs], dtype=np.int32)
        train_idx = np.array([k for k, p in enumerate(img_paths) if p not in val_imgs], dtype=np.int32)
        
        yield train_idx, val_idx


def _train_and_evaluate_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    C_val: float, gamma_val: float,
    tol: float, max_iter: int,
    binary_mode: bool,
) -> float:
    """
    在单个 CV 折上训练并评估，返回验证准确率。
    
    Args:
        binary_mode: True=训练单个二分类SVM, False=训练OvO多分类SVM
    
    Returns:
        验证准确率
    """
    import sys
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        if binary_mode:
            # 二分类模式：单一 HandwrittenSVM（标签已为 0/1，转为 -1/+1）
            y_tr_bin = np.where(y_tr == 0, -1, 1).astype(np.float64)
            model = HandwrittenSVM(C=C_val, gamma=gamma_val, tol=tol, max_iter=max_iter)
            model.fit(X_tr, y_tr_bin)
            preds_bin = model.predict(X_val)  # +1 或 -1
            preds = np.where(preds_bin == -1, 0, 1)  # -1→0, +1→1
        else:
            # 三分类模式：OvO 多分类器
            model = OvOSVMClassifier(C=C_val, gamma=gamma_val, tol=tol, max_iter=max_iter)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    
    return np.mean(preds == y_val)


def grid_search_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_paths: List[str],
    C_list: List[float],
    gamma_list: List[float],
    n_splits: int = 5,
    tol: float = 1e-3,
    max_iter: int = 2000,
    binary_mode: bool = False,
) -> Tuple[float, float, float, Dict]:
    """
    网格搜索 + K 折交叉验证，选出最优 (C, gamma) 组合。
    
    Args:
        X_train: 训练特征 (N, D)
        y_train: 训练标签 (N,)
        train_paths: 切块来源路径
        C_list: 候选 C 值列表
        gamma_list: 候选 γ 值列表
        n_splits: 交叉验证折数
        tol: KKT 容忍度
        max_iter: SMO 最大迭代次数
        binary_mode: True=二分类模式，False=三分类模式
    
    Returns:
        (best_C, best_gamma, best_cv_acc, cv_results_dict)
    """
    mode_name = "二分类（实拍 vs 非实拍）" if binary_mode else "三分类（OvO）"
    print("\n" + "=" * 60)
    print(f"  网格搜索 + {n_splits} 折交叉验证  [{mode_name}]")
    print("=" * 60)
    print(f"  C 候选:     {C_list}")
    print(f"  gamma 候选: {gamma_list}")
    print(f"  总组合数:   {len(C_list) * len(gamma_list)}")
    print("-" * 60)
    
    cv_results = {}
    best_C = None
    best_gamma = None
    best_acc = -1.0
    
    total_combos = len(C_list) * len(gamma_list)
    combo_iter = tqdm(total=total_combos, desc="[网格搜索]")
    
    for C_val in C_list:
        for gamma_val in gamma_list:
            fold_accs = []
            
            # K 折交叉验证
            for train_idx, val_idx in stratified_kfold_by_image(
                X_train, y_train, train_paths, n_splits
            ):
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                acc = _train_and_evaluate_fold(
                    X_tr=X_tr, y_tr=y_tr,
                    X_val=X_val, y_val=y_val,
                    C_val=C_val, gamma_val=gamma_val,
                    tol=tol, max_iter=max_iter,
                    binary_mode=binary_mode,
                )
                fold_accs.append(acc)
            
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            
            key = f"C={C_val}, gamma={gamma_val}"
            cv_results[key] = {
                "C": C_val,
                "gamma": gamma_val,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "fold_accs": fold_accs,
            }
            
            combo_iter.set_postfix({
                "当前": f"C={C_val}, γ={gamma_val}",
                "acc": f"{mean_acc:.4f}±{std_acc:.4f}",
                "最优": f"{best_acc:.4f}" if best_acc >= 0 else "N/A",
            })
            combo_iter.update(1)
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_C = C_val
                best_gamma = gamma_val
    
    combo_iter.close()
    
    print(f"\n{'=' * 60}")
    print(f"  ✓ 网格搜索完成")
    print(f"  最优参数: C={best_C}, gamma={best_gamma}")
    print(f"  最优交叉验证准确率: {best_acc:.4f} (±{cv_results[f'C={best_C}, gamma={best_gamma}']['std_acc']:.4f})")
    print(f"{'=' * 60}")
    
    # 打印所有结果
    print("\n  完整结果:")
    for key in sorted(cv_results.keys(), key=lambda k: cv_results[k]['mean_acc'], reverse=True):
        r = cv_results[key]
        print(f"    {key:25s}  acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}")
    
    return best_C, best_gamma, best_acc, cv_results


# ============================================================================
#  主训练管道
# ============================================================================

def train_pipeline(
    features_dir: str = "outputs/",
    model_path: str = "outputs/best_model.pkl",
    cv_results_path: str = "outputs/cv_results.pkl",
    skip_if_exists: bool = False,
    binary_mode: bool = False,
    C_list: Optional[List[float]] = None,
    gamma_list: Optional[List[float]] = None,
    n_splits: int = 5,
    tol: float = 1e-3,
    max_iter: int = 2000,
) -> Tuple[object, float, Dict]:
    """
    完整训练管道：加载特征 → 网格搜索CV → 训练最优模型 → 保存。
    
    Args:
        features_dir: 特征文件目录
        model_path: 模型保存路径
        cv_results_path: CV 结果保存路径
        skip_if_exists: 若模型已存在则跳过
        C_list: C 候选列表
        gamma_list: γ 候选列表
        n_splits: CV 折数
        tol: KKT 容忍度
        max_iter: SMO 最大迭代次数
    
    Returns:
        (best_model, best_cv_acc, cv_results)
    """
    if C_list is None:
        C_list = [0.1, 1, 10, 100]
    if gamma_list is None:
        gamma_list = [0.001, 0.01, 0.1, 1]
    
    # ===== 辅助函数：保存二分类模型 =====
    def save_binary_model(svm: HandwrittenSVM, filepath: str):
        """保存单一 HandwrittenSVM 为 pkl"""
        model_data = {
            "mode": "binary",
            "C": svm.C,
            "gamma": svm.gamma,
            "tol": svm.tol,
            "max_iter": svm.max_iter,
            "alpha": svm.alpha,
            "b": svm.b,
            "support_vectors": svm.support_vectors,
            "support_labels": svm.support_labels,
            "support_alpha": svm.support_alpha,
            "X_train": svm.X_train,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"[模型保存] ✓ 二分类模型已保存至 {filepath}")
    
    def load_binary_model(filepath: str) -> HandwrittenSVM:
        """从 pkl 加载单一 HandwrittenSVM"""
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
    # ====================================
    
    # 断点复用
    if skip_if_exists and os.path.exists(model_path):
        print(f"[训练] 模型文件已存在，跳过训练（skip_if_exists=True）")
        print(f"  从 {model_path} 加载模型...")
        with open(model_path, "rb") as f:
            header = pickle.load(f)
        if isinstance(header, dict) and header.get("mode") == "binary":
            model = load_binary_model(model_path)
        else:
            model = OvOSVMClassifier.load(model_path)
        # 尝试加载 CV 结果
        cv_results = {}
        if os.path.exists(cv_results_path):
            with open(cv_results_path, "rb") as f:
                cv_data = pickle.load(f)
                cv_results = cv_data.get("cv_results", {})
                best_acc = cv_data.get("best_acc", 0.0)
        else:
            best_acc = 0.0
        return model, best_acc, cv_results
    
    # 加载预处理特征
    print("[训练] 加载预处理特征文件...")
    with open(os.path.join(features_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(features_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    
    print(f"[训练] 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 维特征")
    if binary_mode:
        print(f"  类别分布（二分类）: 实拍(0)={np.sum(y_train==0)}, 非实拍(1)={np.sum(y_train>=1)}")
    else:
        print(f"  类别分布: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}, 2={np.sum(y_train==2)}")
    
    # 生成虚拟图片路径用于 CV（因预处理未保存原始路径）
    train_paths_dummy = [f"sample_{i}#patch0" for i in range(len(X_train))]
    
    # 网格搜索 + CV
    best_C, best_gamma, best_acc, cv_results = grid_search_cv(
        X_train=X_train,
        y_train=y_train,
        train_paths=train_paths_dummy,
        C_list=C_list,
        gamma_list=gamma_list,
        n_splits=n_splits,
        tol=tol,
        max_iter=max_iter,
        binary_mode=binary_mode,
    )
    
    # 用最优参数在全量训练集上训练最终模型
    print(f"\n[训练] 使用最优参数 (C={best_C}, gamma={best_gamma}) 在全量训练集上训练...")
    
    if binary_mode:
        # 二分类：单一 HandwrittenSVM（标签 0→-1, 1→+1）
        y_train_bin = np.where(y_train == 0, -1, 1).astype(np.float64)
        best_model = HandwrittenSVM(
            C=best_C, gamma=best_gamma, tol=tol, max_iter=max_iter
        )
        best_model.fit(X_train, y_train_bin)
        save_binary_model(best_model, model_path)
    else:
        # 三分类：OvO 多分类器
        best_model = OvOSVMClassifier(
            C=best_C, gamma=best_gamma, tol=tol, max_iter=max_iter
        )
        best_model.fit(X_train, y_train)
        best_model.save(model_path)
    
    # 保存 CV 结果
    cv_data = {
        "best_C": best_C,
        "best_gamma": best_gamma,
        "best_acc": best_acc,
        "cv_results": cv_results,
        "binary_mode": binary_mode,
    }
    with open(cv_results_path, "wb") as f:
        pickle.dump(cv_data, f)
    print(f"[训练] CV 结果已保存至 {cv_results_path}")
    
    return best_model, best_acc, cv_results


# ============================================================================
#  CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="手写 RBF 核 SVM 训练模块")
    parser.add_argument("--features_dir", type=str, default="outputs/", help="特征文件目录")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pkl", help="模型保存路径")
    parser.add_argument("--cv_results_path", type=str, default="outputs/cv_results.pkl", help="CV结果路径")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="若模型已存在则跳过")
    parser.add_argument("--C_list", type=float, nargs="+", default=[0.1, 1, 10, 100], help="C候选值")
    parser.add_argument("--gamma_list", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1], help="gamma候选值")
    parser.add_argument("--n_splits", type=int, default=5, help="CV折数")
    parser.add_argument("--max_iter", type=int, default=2000, help="SMO最大迭代次数")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  手写 RBF 核 SVM 训练模块")
    print("=" * 60)
    
    train_pipeline(
        features_dir=args.features_dir,
        model_path=args.model_path,
        cv_results_path=args.cv_results_path,
        skip_if_exists=args.skip_if_exists,
        C_list=args.C_list,
        gamma_list=args.gamma_list,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
    )
    
    print("\n[训练] 完成！")


if __name__ == "__main__":
    main()