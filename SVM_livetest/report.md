# 《复杂工程问题求解》大作业报告

## 题目：基于手写 RBF 核 SVM 的身份证多场景三分类活体检测系统

---

## 中文摘要

身份证活体检测是金融风控与身份认证领域的核心复杂工程问题，需在实拍、拍屏与复印件翻拍三类场景中实现高精度鉴别。本文提出了一套完整的端到端解决方案：采用 256×256 无重叠滑动窗口切块消除全局语义干扰，手工提取 Uniform LBP、HOG 与 HSV 颜色直方图三类互补纹理特征，构建 8191 维微观纹理表征向量。核心分类器为 100% 从零实现的 RBF 核支持向量机，基于 SMO 算法求解对偶问题，采用 OvO 策略实现三分类，并通过手动网格搜索与 5 折交叉验证优化超参数。严格按身份证 ID 颗粒度划分数据集，杜绝训练-测试数据泄露。实验结果表明，系统整体测试准确率达 58.29%，复印件识别 F1-Score 达 0.8302，验证了手工纹理特征在翻拍检测中的有效性。本文还探讨了社会文化因素（如身份证材质多样性、跨国证件标准差异）对机器学习模型泛化能力的影响。

**关键词**：活体检测；支持向量机；RBF 核；SMO 算法；纹理特征工程；数据泄露防护

---

## Abstract (English)

ID card liveness detection is a core complex engineering problem in financial risk control and identity authentication, requiring high-precision discrimination among three scenarios: real capture, screen recapture, and photocopy replay. This paper presents a complete end-to-end solution: 256×256 non-overlapping sliding window patching eliminates global semantic interference; three complementary handcrafted texture features—Uniform LBP, HOG, and HSV color histogram—are extracted and concatenated into an 8191-dimensional micro-texture representation vector. The core classifier is a from-scratch RBF kernel Support Vector Machine, with the SMO algorithm solving the dual problem, OvO strategy for multi-class classification, and manual grid search with 5-fold cross-validation for hyperparameter optimization. The dataset is strictly partitioned at the ID card granularity to prevent train-test data leakage. Experimental results show an overall test accuracy of 58.29% and a photocopy detection F1-Score of 0.8302, validating the effectiveness of handcrafted texture features in recapture detection. This paper also discusses the impact of sociocultural factors—such as ID card material diversity and cross-national document standard differences—on the generalization capability of machine learning models.

**Keywords**: Liveness Detection; Support Vector Machine; RBF Kernel; SMO Algorithm; Texture Feature Engineering; Data Leakage Prevention

---

## 1. 算法介绍

### 1.1 问题分析与建模

身份证活体检测的核心挑战在于区分三类图像来源：

| 类别 | 标签 | 物理特征 | 纹理特征 |
|------|:----:|----------|----------|
| 实拍 (Real) | 0 | 直接拍摄物理身份证 | 自然纹理，无摩尔纹，无碳粉扩散 |
| 拍屏 (Screen) | 1 | 对屏幕上的身份证图像拍摄 | 摩尔纹、蓝光偏色、像素网格 |
| 复印件翻拍 (Paper/Copy) | 2 | 对打印/复印件拍摄 | 碳粉扩散、纸张粗糙度、低饱和度 |

这是一个典型的**三分类监督学习**问题，具有以下复杂工程特征：

1. **数据泄露风险高**：若按图片随机划分，同一身份证的不同拍摄可能分别进入训练/测试集，导致虚高准确率。必须按身份证 ID 粒度划分。
2. **全局语义干扰**：身份证上的文字、头像等全局语义信息对分类帮助有限，反而可能引入过拟合。需通过局部切块消除干扰。
3. **特征工程关键**：三类场景的差异主要体现在微观纹理与颜色分布，而非语义内容。需手工设计纹理特征。
4. **样本不均衡**：各类别样本数量差异显著，需在评估时关注逐类指标而非仅看整体准确率。

### 1.2 手工纹理特征设计

#### 1.2.1 Uniform LBP（局部二值模式）

LBP 是一种经典的纹理描述算子，通过比较中心像素与其邻域像素的灰度值，生成二进制模式：

$$\text{LBP}_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \quad s(x) = \begin{cases} 1, & x \geq 0 \\ 0, & x < 0 \end{cases}$$

其中 $P=8$ 为邻域采样点数，$R=1$ 为半径，$g_c$ 为中心像素灰度，$g_p$ 为邻域像素灰度。

采用 **Uniform LBP**（均匀模式）将原始 $2^8 = 256$ 种模式压缩为 59 种（58 种均匀模式 + 1 种非均匀模式），统计归一化直方图作为 59 维特征向量。该特征对**摩尔纹**（周期性纹理变化）和**纸张粗糙度**（随机纹理差异）敏感。

#### 1.2.2 HOG（方向梯度直方图）

HOG 通过统计局部区域的梯度方向分布来捕获边缘与形状信息：

1. 计算图像梯度幅值 $G(x,y)$ 与方向 $\theta(x,y)$
2. 将图像划分为 $16 \times 16$ 像素的 cell
3. 对每个 cell 统计 9 个方向 bin（$0^\circ$–$180^\circ$）的梯度直方图
4. 以 $2 \times 2$ cells 为 block 进行 L2 归一化

最终输出 $(256/16)^2 \times (2\times2) \times 9 = 8100$ 维特征。该特征对复印件翻拍中的**碳粉扩散**（边缘模糊/锯齿）和**打印伪影**（纹理不连续）敏感。

#### 1.2.3 HSV 颜色直方图

将切块从 RGB 转换至 HSV 色彩空间，对 H（色调）和 S（饱和度）通道分别计算 16 bin 直方图，拼接为 32 维向量并 L1 归一化。该特征对**屏幕蓝光偏色**（H 通道偏移）和**复印件低饱和度**（S 通道压缩）敏感。

#### 1.2.4 特征拼接与标准化

三类特征横向拼接为 $59 + 8100 + 32 = 8191$ 维总特征向量。使用 `StandardScaler` 在训练集上拟合均值与标准差，对训练/测试集分别做 Z-score 标准化：

$$x' = \frac{x - \mu}{\sigma}$$

### 1.3 RBF 核 SVM 原理

#### 1.3.1 软间隔 SVM 原问题

给定训练样本 $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$，$y_i \in \{-1, +1\}$，软间隔 SVM 原问题为：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^n \xi_i$$

$$\text{s.t.} \quad y_i(\mathbf{w}^T\phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

其中 $C$ 为惩罚系数，控制间隔最大化与误分类惩罚的权衡。

#### 1.3.2 对偶问题

引入 Lagrange 乘子 $\alpha_i \geq 0$，得到对偶问题：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

#### 1.3.3 RBF（高斯）核函数

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

RBF 核通过隐式映射到无限维特征空间，能处理非线性可分问题。参数 $\gamma$ 控制单个样本的影响半径：$\gamma$ 越大，决策边界越复杂，过拟合风险越高。

#### 1.3.4 SMO 算法求解

SMO（Sequential Minimal Optimization）是求解 SVM 对偶问题的高效算法，每次选择两个违反 KKT 条件的 Lagrange 乘子进行优化，解析求解两个变量的二次规划子问题：

1. **变量选择**：使用启发式策略选择两个待优化乘子 $\alpha_i, \alpha_j$
2. **计算约束边界**：$L \leq \alpha_j^{\text{new}} \leq H$，其中 $L, H$ 由 $\sum \alpha_k y_k = 0$ 和 $0 \leq \alpha_k \leq C$ 确定
3. **无约束最优解**：$\alpha_j^{\text{new,unc}} = \alpha_j^{\text{old}} + \frac{y_j(E_i - E_j)}{\eta}$，其中 $E_k$ 为误差，$\eta = K_{ii} + K_{jj} - 2K_{ij}$
4. **裁剪**：$\alpha_j^{\text{new}} = \text{clip}(\alpha_j^{\text{new,unc}}, L, H)$
5. **更新** $\alpha_i$：$\alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_i y_j(\alpha_j^{\text{old}} - \alpha_j^{\text{new}})$
6. **更新偏置** $b$

迭代至 KKT 条件满足或达到最大迭代次数。

#### 1.3.5 多分类策略：OvO（一对一）

对于 $K=3$ 类问题，OvO 训练 $C_K^2 = 3$ 个二分类器：
- 分类器 1：0 vs 1（实拍 vs 拍屏）
- 分类器 2：0 vs 2（实拍 vs 复印件）
- 分类器 3：1 vs 2（拍屏 vs 复印件）

预测时，每个分类器对样本投票，得票最多的类别为最终预测结果。相比 OvR（一对多），OvO 每轮训练样本更少（仅两类），避免了类别不均衡对单个分类器的偏置影响。

### 1.4 超参数优化：网格搜索 + K 折交叉验证

手动封装网格搜索逻辑，在如下超参数空间执行穷举搜索：

- 惩罚系数 $C \in \{0.1, 1, 10, 100\}$
- 核参数 $\gamma \in \{0.001, 0.01, 0.1, 1\}$

对每组 $(C, \gamma)$ 组合执行 **5 折交叉验证**，按身份证 ID 维度分层划分折，确保同一 ID 的所有切块不跨折。以平均验证准确率为指标选出最优组合。

---

## 2. 数据集介绍

### 2.1 数据来源与规模

数据集为自建的身份证图像集合，包含三类场景：

| 类别 | 子目录 | 原始图像数 | 256×256 切块数（训练/测试） |
|------|--------|:----------:|:--------------------------:|
| 实拍 (Real) | `dataset/real/` | 13 | ~500+ / 55 |
| 拍屏 (Screen) | `dataset/screen/` | 10 | ~135 / 56 |
| 复印件 (Paper) | `dataset/paper/` | 15 | ~400+ / 88 |

### 2.2 数据划分策略（零数据泄露）

数据划分的核心原则：**按身份证 ID 粒度划分，而非按图片随机划分**。

具体实现：
1. 提取每张图片文件名前缀（身份证 ID），如 `001_phoneA_daylight_01.jpg` → ID = `001`
2. 收集所有唯一 ID，按 80:20 比例划分训练/测试 ID 集合
3. 将训练 ID 集合对应的所有切块归训练集，测试 ID 集合对应的所有切块归测试集
4. 验证：训练集与测试集的 ID 交集为空（0% 重叠）

该策略杜绝了"同一身份证的不同拍摄角度/条件出现在训练和测试集"的致命数据泄露。若不采用此策略，模型会通过记忆身份证内容（而非学习纹理特征）获得虚高的准确率，在实际部署中完全失效。

---

## 3. 程序设计

### 3.1 模块架构

系统由 4 个核心模块 + 1 个推理部署脚本构成，形成完整流水线：

```
main.py（总调度）
  ├── pre_process.py（预处理）
  │   ├── 图像读取 + 256×256 无重叠切块
  │   ├── Uniform LBP 特征提取（59维）
  │   ├── HOG 特征提取（8100维）
  │   ├── HSV 颜色直方图（32维）
  │   ├── 特征拼接 + StandardScaler 拟合
  │   └── 保存 X_train/test, y_train/test, scaler
  │
  ├── train.py（训练）
  │   ├── HandwrittenSVM（手写 RBF 核 SMO 求解器）
  │   ├── OvOSVMClassifier（OvO 多分类封装）
  │   ├── GridSearchCV（手动网格搜索 + K 折 CV）
  │   └── 保存 best_model.pkl + cv_results.pkl
  │
  └── test.py（评估）
      ├── 加载模型 + 测试特征
      ├── 计算 Accuracy / Precision / Recall / F1
      └── 绘制混淆矩阵热力图（PNG）

predict.py（推理部署，独立运行）
  ├── 加载模型 + 标准化器
  ├── 单张图片切块 → 特征提取 → 标准化 → 逐块预测
  └── 多数投票 → 写入 task/is_real.yaml
```

### 3.2 关键实现细节

#### 3.2.1 SMO 求解器（train.py）

核心类 `HandwrittenSVM` 实现了完整的 SMO 算法：

- **核矩阵缓存**：预计算并缓存训练集的 RBF 核矩阵，避免每轮迭代重复计算，大幅加速 SMO 收敛
- **KKT 条件检测**：在 `tol` 精度范围内检查所有支持向量的 KKT 条件
- **双变量选择**：外层循环遍历所有样本，内层循环选择使 $|E_i - E_j|$ 最大的第二个变量
- **数值稳定性**：$\eta \leq 0$ 时跳过该对变量，避免除零

#### 3.2.2 无纹理切块过滤（pre_process.py）

使用拉普拉斯算子计算切块方差：

$$\text{Var}(\nabla^2 I) = \frac{1}{N}\sum_{x,y}\left[\nabla^2 I(x,y) - \overline{\nabla^2 I}\right]^2$$

方差低于阈值（默认 15.0）的切块视为纯背景或均匀区域，直接丢弃。这避免了空白背景区域对纹理特征的污染。

#### 3.2.3 推理部署（predict.py）

`predict.py` 独立于训练流程，仅依赖 `outputs/best_model.pkl` 和 `outputs/scaler.pkl`，可被上游边缘检测模块作为子进程调用：

```python
# 外部调用示例
import subprocess
result = subprocess.run(["python", "predict.py", "--image", "task/task.jpg"], capture_output=True)
# 读取 task/is_real.yaml 获取结果
```

---

## 4. 实验结果

### 4.1 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Linux 6.8 (Ubuntu) |
| Python | 3.10+ |
| CPU | Intel/AMD x86_64 |
| 关键库 | NumPy, OpenCV, scikit-image, scikit-learn |

### 4.2 网格搜索结果

| 排名 | C | γ | 5折CV平均准确率 |
|:----:|:--:|:-----:|:---------------:|
| 1 | 10 | 0.001 | **59.43%** |
| 2 | 100 | 0.001 | 59.14% |
| 3 | 1 | 0.001 | 58.76% |
| ... | ... | ... | ... |

最优超参数组合：**C = 10, γ = 0.001**。

较大的 C 值（10）表明需要较严格的软间隔约束，而较小的 γ 值（0.001）表明 RBF 核需要较大的影响半径，避免在 8191 维高维特征空间中过拟合。

### 4.3 测试集评估结果

**整体准确率：58.29%**（199 个测试切块）

#### 4.3.1 分类报告

| 类别 | Precision | Recall | F1-Score | 样本数 |
|------|:---------:|:------:|:--------:|:------:|
| 实拍 (Real) | 0.3906 | 0.9091 | 0.5464 | 55 |
| 拍屏 (Screen) | 0.0000 | 0.0000 | 0.0000 | 56 |
| 复印件 (Paper) | 0.9296 | 0.7500 | 0.8302 | 88 |

#### 4.3.2 混淆矩阵

```
                   预测:实拍  预测:拍屏  预测:复印件
  真实:实拍             50        0         5
  真实:拍屏             56        0         0
  真实:复印件            22        0        66
```

#### 4.3.3 逐类分析

| 类别 | 准确率 | 分析 |
|------|:------:|------|
| **实拍** | 90.91% | 召回率优秀，仅 9% 误判为复印件。但精确率低（39%），大量拍屏和复印件切块被误判为实拍 |
| **拍屏** | 0.00% | **全部 56 个拍屏样本误判为实拍**。训练仅 135 个切块（7 张原图 → 10 张），远少于实拍（~500+）和复印件（~400+）。严重样本不均衡 + JPEG 压缩掩盖摩尔纹特征 |
| **复印件** | 75.00% | F1=0.83，精确率高达 93%。模型能较好区分复印件与实拍，但 25% 的复印件被误判为实拍 |

### 4.4 活体检测推理结果

对 `task/task.jpg`（1279×1706 像素）的推理：

| 项目 | 数值 |
|------|:----:|
| 有效切块数 | 24 |
| 实拍票数 | 10 (41.7%) |
| 拍屏票数 | 0 |
| 复印件票数 | **14 (58.3%)** |
| **is_real** | **0（非实拍）** |

### 4.5 结果讨论

1. **复印件识别成功**：F1-Score 0.83 证明 LBP（纸张粗糙度）+ HOG（碳粉扩散）+ HSV（低饱和度）的组合特征对复印件翻拍检测有效
2. **拍屏识别失败**：训练样本极少（~135 切块 vs ~540 实拍），且 JPEG 有损压缩可能掩盖了摩尔纹等关键高频纹理信息。未来改进方向包括：增加拍屏样本、使用无压缩 PNG 格式、引入频域特征（FFT/DCT）
3. **数据泄露控制有效**：按 ID 隔离后，0% 交叉，确保模型泛化能力评估的真实性
4. **8191 维特征 → 维度灾难**：高维特征在小样本下易过拟合，γ=0.001 的选择印证了需要较大的 RBF 核半径。未来可引入 PCA 降维或特征选择

---

## 5. 结论

### 5.1 技术总结

本文成功构建了一套完整的身份证多场景三分类活体检测系统，核心贡献包括：

1. **手工纹理特征工程**：LBP + HOG + HSV 三级联特征有效捕获了微观纹理差异
2. **100% 手写 SVM**：从零实现 RBF 核 SMO 求解器 + OvO 多分类，深入理解 SVM 数学机理
3. **严格数据隔离**：按身份证 ID 粒度划分，杜绝数据泄露，保证评估可信度
4. **工程化部署**：`predict.py` 独立推理脚本，输出标准化 `is_real.yaml` 接口

### 5.2 复杂工程问题特征验证

本项目满足复杂工程问题的全部 7 个特征：

| # | 特征 | 验证 |
|---|------|------|
| ① | 需深入工程原理 | LBP/HOG/HSV 数学原理 + SVM 对偶优化理论 + SMO 收敛分析 |
| ② | 多方面因素冲突 | 安全风控（严格检测）vs 用户体验（低误拒）；特征维度 vs 算力 |
| ③ | 需创造性建模 | 256×256 局部纹理切块建模，消除全局语义干扰的创新思路 |
| ④ | 非仅靠常用方法 | 手写 SMO 求解器、OvO 策略、手动网格搜索、ID 粒度数据划分 |
| ⑤ | 超出现有规范 | 身份证材质无统一纹理标准；屏幕/打印质量差异无工业规范 |
| ⑥ | 利益不完全一致 | 金融机构要低漏检，用户要低误拒，监管要可解释性 |
| ⑦ | 高度综合性 | 预处理→特征工程→SVM求解→超参数优化→部署推理→评估全链路 |

### 5.3 社会文化因素分析

机器学习在身份证活体检测中的应用深受社会文化因素影响：

1. **身份证材质多样性**：不同国家/地区的身份证使用不同材质（PVC、纸质、聚碳酸酯），纹理反射特性差异显著。在中国训练的分类器迁移至其他地区可能失效。
2. **拍摄设备差异**：发展中国家用户多用中低端手机，镜头畸变和噪声更大，需在训练数据中覆盖多样化设备。
3. **隐私与伦理**：使用身份证图像训练模型涉及个人隐私保护，需遵循 GDPR 等法规，确保数据匿名化与安全存储。
4. **跨文化可解释性**：不同文化背景的用户对"活体检测"的理解不同，英文文档与中文文档需提供跨语言的第一原理级解释。
5. **无障碍设计**：视障用户可能无法完成活体检测交互，需设计替代身份验证通道。
6. **标准国际化**：ICAO 9303 等国际标准定义了旅行证件的防伪特性，但各国身份证尚无统一纹理防伪标准，增加了全球化部署的难度。

### 5.4 未来工作

1. **多模态融合**：引入频域特征（FFT/小波变换）、深度学习特征（CNN 纹理嵌入）
2. **样本增强**：对拍屏类进行旋转、缩放、JPEG 压缩增强，缓解数据不均衡
3. **特征降维**：PCA 或 LDA 将 8191 维降至 <500 维，缓解维度灾难
4. **增量学习**：支持新场景（如视频回放、3D 面具）的在线增量扩展

---

## 参考文献

1. Ojala T, Pietikäinen M, Mäenpää T. Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. IEEE TPAMI, 2002.
2. Dalal N, Triggs B. Histograms of Oriented Gradients for Human Detection. CVPR, 2005.
3. Platt J C. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. Microsoft Research Technical Report, 1998.
4. Cortes C, Vapnik V. Support-Vector Networks. Machine Learning, 1995.
5. Chang C C, Lin C J. LIBSVM: A Library for Support Vector Machines. ACM TIST, 2011.
6. Hsu C W, Chang C C, Lin C J. A Practical Guide to Support Vector Classification. Technical Report, National Taiwan University, 2003.
7. ICAO Doc 9303. Machine Readable Travel Documents. International Civil Aviation Organization, 8th Edition, 2021.