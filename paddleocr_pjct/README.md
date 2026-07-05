# PaddleOCR 身份证检测模型微调项目

基于 **PaddleOCR PP-OCRv4** 对身份证图片进行**文本检测**（Detection）的微调训练项目。

> **系统要求：** 本教程仅适用于 **Windows 10/11**。所有命令请使用 **Anaconda Prompt**（不要用 PowerShell）执行。

---

## 目录结构

```
paddleocr_pjct/
├── README.md                          # 本文件（项目说明）
├── requirements.txt                   # Python 依赖清单
├── data/                              # 数据集目录
│   ├── train/                         #   训练集（200 张）
│   │   ├── id_card_0000.jpg
│   │   ├── ...
│   │   └── label.txt                  #   训练标注
│   └── val/                           #   验证集（50 张）
│       ├── id_card_0000.jpg
│       ├── ...
│       └── label.txt                  #   验证标注
├── scripts/                           # 脚本目录
│   ├── init_check.py                  #   环境检查脚本
│   ├── generate_data.py               #   身份证数据生成脚本
│   └── train_launcher.bat             #   训练启动脚本（Windows）
├── PaddleOCR/                         # PaddleOCR 源码
│   ├── configs/det/PP-OCRv4/
│   │   └── PP-OCRv4_mobile_det.yml    #   检测模型配置文件
│   └── tools/
│       └── train.py                   #   训练入口
└── output/                            # 训练产出
    └── PP-OCRv4_mobile_det/           #   模型保存目录
        ├── iter_epoch_10.pdparams     #   epoch 10 权重
        ├── iter_epoch_20.pdparams     #   epoch 20 权重
        ├── ...
        ├── iter_epoch_50.pdparams     #   epoch 50 权重（最终）
        ├── latest.pdparams            #   最新权重
        └── train.log                  #   训练日志
```

---

## 第 1 步：安装依赖环境

### 1.1 前置条件

| 软件 | 版本要求 | 检查方法 |
|------|---------|---------|
| **CUDA** | 仅支持 11.8（GPU 用户） | `nvcc --version` |
| **Python** | 3.9 ~ 3.12 | `python --version` |
| **pip** | 21.0 及以上 | `pip --version` |
| **Anaconda / Miniconda** | 任意版本 | `conda --version` |

> **没有 GPU？** 可将 `requirements.txt` 中的 `paddlepaddle-gpu` 改为 `paddlepaddle`（CPU 版），训练速度会慢很多但功能完全一致。
>
> **Windows GPU 用户重要提示：** PaddlePaddle Windows 版 GPU 仅支持 **CUDA 11.8**，不支持 CUDA 12.x。如果你的显卡驱动是 CUDA 12.x，有两个选择：
> 1. 使用 CPU 版：将 `requirements.txt` 中的 `paddlepaddle-gpu` 改为 `paddlepaddle`
> 2. 降级安装 CUDA 11.8（不推荐，比较麻烦）

**检查你的 CUDA 版本：**

打开命令提示符（CMD），输入：

```batch
nvcc --version
```

- 如果显示 `not recognized`，说明没有装 CUDA 工具包，但不影响——conda 会在环境中自动安装所需版本。
- 如果显示 `release 11.8`，可以正常使用 GPU 版。
- 如果显示 `release 12.x`，建议改用 CPU 版。

### 1.2 创建 conda 虚拟环境

打开 **Anaconda Prompt**，输入以下命令：

```batch
# 创建名为 mchlrn 的虚拟环境，Python 版本为 3.10
conda create -n mchlrn python=3.10 -y

# 激活环境
conda activate mchlrn
```

> **注意：** 后续所有命令都需要在 `mchlrn` 环境下运行。如果你新开了一个终端窗口，需要再次执行 `conda activate mchlrn` 激活环境。
>
> **确认已激活：** 终端提示符最左边应该显示 `(mchlrn)`，像这样：`(mchlrn) C:\Users\...>`

### 1.3 安装 Python 依赖

> ⚠️ **首先确保你在项目目录下！** 在终端中输入以下命令切换到项目目录：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```
>
> 输入 `dir` 确认能看到 `requirements.txt` 文件。

确认当前目录正确后，安装依赖：

```batch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 使用清华镜像源加速下载，耗时约 3~10 分钟（视网络情况而定）。
>
> 如果安装过程中某个包下载失败，可以重新运行上述命令——已下载的包不会重复下载。

**安装完成后验证一下：**

```batch
pip show paddlepaddle-gpu
```

如果显示 `Name: paddlepaddle-gpu`、`Version: 2.6.2` 等信息，说明核心框架安装成功。

> 如果这里报错说找不到，说明 `paddlepaddle-gpu` 没装上。常见原因是你的 CUDA 版本不是 11.8，请将 `requirements.txt` 中的 `paddlepaddle-gpu==2.6.2` 改为 `paddlepaddle==2.6.2`（CPU 版），然后重新运行 `pip install`。

### 1.4 安装 cuDNN（GPU 版本必需）

```batch
conda install -c conda-forge cudnn=9.2 -y
```

> cuDNN 是 NVIDIA 的深度学习加速库，PaddlePaddle GPU 版需要它。conda 会自动处理版本兼容问题。
>
> 安装后，cuDNN 的 DLL 文件位于 `%CONDA_PREFIX%\Library\bin\`。**如果后续启动训练时报 DLL 错误**，手动执行一次：
>
> ```batch
> set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
> ```
>
> 或者直接使用项目自带的 `scripts\train_launcher.bat` 启动训练（已自动配置）。

### 1.5 验证环境

> ⚠️ **确保你在项目目录下！** 如果不是，先执行：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```

运行环境检查脚本：

```batch
python scripts\init_check.py
```

**成功输出示例：**

```
==================================================
  PaddleOCR 初始化环境检查
==================================================
✅ PaddleOCR 导入成功！
   PaddleOCR 版本: 3.5.0
   PaddlePaddle 版本: 2.6.2
   GPU 可用: ✅ (CUDA 版本: 11.8)
==================================================
  ✅ 环境检查通过，可以开始开发！
==================================================
```

如果看到 `GPU 可用: ❌ (使用 CPU 模式)`，说明你安装的是 CPU 版 PaddlePaddle——功能正常，只是训练慢。

如果看到 `❌ PaddleOCR 导入失败`，请回到第 1.3 步检查依赖是否正确安装。

---

## 第 2 步：生成训练数据

本项目包含一个身份证模拟数据生成器，可以为每张图片自动生成精确的 PaddleOCR 格式标注。

### 2.1 中文字体检查

生成身份证图片需要使用中文字体。脚本 `scripts\generate_data.py` 已内置字体自动检测，会按以下顺序搜索：

1. `C:\Windows\Fonts\msyh.ttc` — 微软雅黑（Windows 自带）
2. `C:\Windows\Fonts\simsun.ttc` — 宋体（Windows 自带）
3. `C:\Windows\Fonts\simhei.ttf` — 黑体（Windows 自带）
4. `C:\Windows\Fonts\msyhbd.ttf` — 微软雅黑粗体（Windows 自带）

> Windows 系统自带这些字体，通常无需额外安装。如果自动检测失败，请参考本文末尾 FAQ 中的字体问题解决方案。

### 2.2 运行数据生成

> ⚠️ **确保你在项目目录下！** 如果不是，先执行：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```

```batch
python scripts\generate_data.py
```

**运行过程示例：**

```
==================================================
  身份证 OCR 训练数据生成
==================================================
  检测到字体: C:\Windows\Fonts\msyh.ttc

📁 生成训练集 (200 张)...
  ✨ 已生成 50/200 张
  ✨ 已生成 100/200 张
  ✨ 已生成 150/200 张
  ✨ 已生成 200/200 张
✅ 数据集生成完成！共 200 张，存放于 data/train

📁 生成验证集 (50 张)...
  ✨ 已生成 50/50 张
✅ 数据集生成完成！共 50 张，存放于 data/val

==================================================
  🎉 全部完成！共生成 250 张身份证图片
  📂 训练集: data/train/
  📂 验证集: data/val/
  📄 标注文件: data/train/label.txt, data/val/label.txt
==================================================
```

### 2.3 生成结果

| 数据集 | 数量 | 图片位置 | 标注位置 |
|--------|------|---------|---------|
| **训练集** | 200 张 | `data\train\` | `data\train\label.txt` |
| **验证集** | 50 张 | `data\val\` | `data\val\label.txt` |

### 2.4 验证生成的数据（可选）

生成完毕后，可以快速检查数据是否完整：

```batch
dir data\train\*.jpg | find /c ".jpg"
dir data\val\*.jpg | find /c ".jpg"
```

应该分别显示 `200` 和 `50`。

看一眼标注文件的格式：

```batch
type data\train\label.txt | more
```

按 `Ctrl+C` 退出查看。

### 2.5 标注格式说明

每张图片的标注为 **PaddleOCR 标准格式**（每行一条记录）：

```
id_card_0000.jpg	[{"transcription": "王伟", "points": [[200,108],[358,108],[358,168],[200,168]]}, ...]
```

一条标注由两部分组成（用 Tab 键分隔）：
- **图片文件名**（左半部分）
- **标注 JSON 数组**（右半部分），每个标注对象包含：
  - `transcription`：文本内容（姓名、性别、身份证号等）
  - `points`：文本框的四个角坐标，格式为 `[[左上], [右上], [右下], [左下]]`

**标注包含的字段：**

| 字段 | 示例 | 说明 |
|------|------|------|
| 姓名 | 王伟 | 从常见中文字库中随机生成 |
| 性别 | 男/女 | 随机选择 |
| 民族 | 汉/蒙古/回/藏/... | 支持 56 个民族随机选择 |
| 出生 | 1990年05月20日 | 从身份证号中自动计算得出 |
| 地址 | 北京市朝阳区中山路36号... | 随机城市+区+街道+门牌号组合 |
| 身份证号 | 110101199005201234 | 18 位，包含正确的校验码 |

---

## 第 3 步：配置并启动训练

### 3.1 配置文件说明

配置文件位置：`PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml`

用记事本或 VS Code 打开这个文件，以下是需要了解的关键配置项：

**训练参数（Global）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `use_gpu` | `True` | 是否使用 GPU。CPU 用户请改为 `False` |
| `epoch_num` | `50` | 训练总轮数。250 张图跑 50 轮约需 20-40 分钟（GPU） |
| `pretrained_model` | URL | 预训练权重下载链接，首次训练会自动下载 |
| `save_model_dir` | `./output/PP-OCRv4_mobile_det` | 模型保存目录 |

**数据配置（Train / Eval）：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `Train.dataset.data_dir` | `./data/train/` | 训练图片目录 |
| `Train.dataset.label_file_list` | `['./data/train/label.txt']` | 训练标注文件 |
| `Eval.dataset.data_dir` | `./data/val/` | 验证图片目录 |
| `Eval.dataset.label_file_list` | `['./data/val/label.txt']` | 验证标注文件 |

**性能相关：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `Train.loader.batch_size_per_card` | `8` | 每张 GPU 的 batch size。如果显存不足，改成 `4` |
| `Train.loader.num_workers` | `8` | 数据加载线程数。CPU 训练时建议改成 `0` 或 `2` |
| `Optimizer.lr.learning_rate` | `0.001` | 初始学习率 |

> 一般情况下**不需要修改任何配置**，直接用默认值即可。如果遇到显存不足或报错，再回来调整。

### 3.2 启动训练

> ⚠️ **确保你在项目目录下！** 如果不是，先执行：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```
>
> 并确认 conda 环境已激活（提示符前缀有 `(mchlrn)`）：
>
> ```batch
> conda activate mchlrn
> ```

**推荐方式：使用启动脚本**

```batch
scripts\train_launcher.bat
```

这个脚本会自动做以下事情：
1. 切换到项目根目录
2. 将 cuDNN 的 DLL 路径添加到 `PATH`
3. 显示训练配置摘要
4. 启动训练

**备选方式：直接运行训练命令**

```batch
python PaddleOCR\tools\train.py -c PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml
```

### 3.3 训练过程解读

训练启动后，首先会下载预训练权重（约 14MB），然后开始训练。

每 100 步输出一次日志，格式如下：

```
epoch:[1/50]  global_step:25  lr:0.000290  loss:7.365881
  loss_shrink_maps:4.857982    # 文本区域收缩图 Loss — 衡量文本框定位精度
  loss_threshold_maps:1.565145 # 阈值图 Loss — 衡量文字边界清晰度
  loss_binary_maps:0.974090    # 二值图 Loss — 衡量文字/非文字分类精度
  ips:16.08 samples/s          # 每秒处理图片数
  eta:0:10:09                  # 预计剩余训练时间
  max_mem_reserved:5777 MB     # GPU 显存占用
```

**Loss 正常下降趋势参考：**

| Epoch | 总 Loss 范围 | 说明 |
|-------|-------------|------|
| 0 ~ 5 | 7.3 → 6.4 | 快速下降阶段，模型在快速学习 |
| 6 ~ 15 | 6.2 → 1.8 | 继续收敛，学习速度放缓 |
| 16 ~ 30 | 1.4 ~ 1.5 | 进入平稳震荡区 |
| 31 ~ 50 | 1.4 → 1.0 | 最终收敛至 ~1.0 附近 |

> **注意：** 如果前 5 个 epoch 的 Loss 完全没有下降，或者 Loss 变成了 `NaN`，请参考 FAQ 中的解决方案。

### 3.4 训练产出

训练完成后，模型保存在 `output\PP-OCRv4_mobile_det\` 目录：

```
output\PP-OCRv4_mobile_det\
├── iter_epoch_10.pdparams    # epoch 10 模型权重（约 14MB）
├── iter_epoch_10.pdopt       # epoch 10 优化器状态（用于恢复训练）
├── iter_epoch_10.states      # epoch 10 训练状态
├── iter_epoch_20.*           # epoch 20
├── iter_epoch_30.*           # epoch 30
├── iter_epoch_40.*           # epoch 40
├── iter_epoch_50.*           # epoch 50（最终模型，推理时用这个）
├── latest.*                  # 最新状态（中断后从此恢复）
├── config.yml                # 训练配置的备份
└── train.log                 # 完整训练日志
```

> **推理时通常使用 `iter_epoch_50`**（最后一个 epoch 的权重），Loss 最低的那个 epoch 效果最好。

---

## 第 4 步：测试模型（推理）

训练完成后，用训练好的模型来检测一张身份证图片，看看效果。

### 4.1 准备测试图片

> ⚠️ **确保你在项目目录下！** 如果不是，先执行：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```

可以使用验证集中的任意图片进行测试：

```batch
dir data\val\id_card_0000.jpg
```

如果有文件信息输出，说明测试图片存在。

### 4.2 命令行推理

```batch
python PaddleOCR\tools\infer_det.py ^
    -c PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml ^
    -o Global.infer_img=data\val\id_card_0000.jpg ^
    -o Global.checkpoints=output\PP-OCRv4_mobile_det\iter_epoch_50
```

> **说明：** 上面命令中 `^` 是 Windows CMD 的续行符，表示这一行命令还没结束。你也可以把整个命令写在一行（去掉 `^`）。

**参数含义：**
- `-c`：指定配置文件
- `-o Global.infer_img`：要检测的图片路径（可以改成你自己的图片）
- `-o Global.checkpoints`：使用的模型权重（去掉 `.pdparams` 后缀）

推理结果会在 `output\` 目录下生成一张带检测框的图片。

### 4.3 用 Python 代码推理

创建一个 `.py` 文件（例如 `test.py`），放在项目根目录：

```python
from paddleocr import PaddleOCR

# 初始化 OCR（使用微调后的检测模型）
ocr = PaddleOCR(
    det_model_dir="output/PP-OCRv4_mobile_det/iter_epoch_50",
    use_angle_cls=False,
    lang="ch",
    use_gpu=True,              # CPU 用户改成 False
)

# 检测图片
result = ocr.ocr("data/val/id_card_0000.jpg", cls=False)

# 打印检测结果
for line in result[0]:
    bbox, (text, score) = line[0], line[1]
    print(f"文本: {text}, 置信度: {score:.4f}, 位置: {bbox}")
```

> ⚠️ **确保你在项目目录下！** 如果不是，先执行：
>
> ```batch
> cd /d C:\Users\27695\Desktop\paddleocr_pjct
> ```

然后在终端中运行：

```batch
python test.py
```

---

## 常见问题（FAQ）

### Q1: 训练时提示 `cudnn64_9.dll` 找不到

**原因：** cuDNN 的 DLL 文件不在系统 PATH 中。

**解决：**

```batch
# 临时解决（当前终端有效，每次新开终端都要执行一次）：
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"

# 永久解决：将以下路径添加到系统环境变量 PATH 中
# 通常是：C:\Users\你的用户名\anaconda3\envs\mchlrn\Library\bin
# 或者是：C:\ProgramData\anaconda3\envs\mchlrn\Library\bin
```

> 也可以直接使用 `scripts\train_launcher.bat` 启动训练，脚本会自动处理这个问题。

### Q2: 安装依赖时某个包反复下载失败

**原因：** 网络不稳定（即使用了清华镜像）。

**解决：**

```batch
# 逐个安装，跳过已装好的包
pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Pillow scipy scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install shapely tqdm PyYAML lmdb rapidfuzz -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 也可以换其他镜像试试：
> - 阿里：`https://mirrors.aliyun.com/pypi/simple/`
> - 中科大：`https://pypi.mirrors.ustc.edu.cn/simple/`

### Q3: 训练时提示 `ModuleNotFoundError: No module named 'xxx'`

**原因：** 某个依赖包没装上，或者你在错误的 conda 环境中。

**解决：**

```batch
# 第 1 步：确认 conda 环境已激活（提示符前有 (mchlrn)）
conda activate mchlrn

# 第 2 步：确认在项目目录下
cd /d C:\Users\27695\Desktop\paddleocr_pjct

# 第 3 步：重新安装所有依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 训练时 `numpy` 相关报错

**原因：** scikit-image 或 scipy 版本与当前 numpy 不兼容。

**解决：**

```batch
pip install --upgrade scikit-image scipy
```

### Q5: 训练 Loss 不下降，或者变成 NaN

**原因：** 学习率过大、数据标注异常或 batch size 不合适。

**解决：**

1. 检查 `data\train\label.txt` 中的坐标是否在图片范围内
2. 降低学习率：打开 `PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml`，找到 `learning_rate`，改为 `0.0001`
3. 在配置文件的 `Global` 部分添加一行：`fix_nan: True`

### Q6: 训练时 `out of memory`（显存不足）

**原因：** batch size 太大或图片分辨率太高。

**解决：** 打开 `PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml`，降低 batch size：

```yaml
Train:
  loader:
    batch_size_per_card: 4    # 原来是 8，改成 4 或 2
```

### Q7: 生成数据时字体报错 "未找到可用的中文字体"

**原因：** 自动字体检测失败。

**解决：**

```batch
# 先确认字体文件是否存在
dir C:\Windows\Fonts\msyh.ttc
dir C:\Windows\Fonts\simsun.ttc
```

如果字体文件存在但仍报错，手动指定字体路径。编辑 `scripts\generate_data.py`，找到第 81 行附近的：

```python
FONT_PATH = detect_font_path()
```

在它下面添加一行，强制指定字体：

```python
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
```

然后重新运行 `python scripts\generate_data.py`。

### Q8: 如何导出为推理模型？

训练得到的模型可以直接用，但如果想部署到生产环境（不依赖训练框架），可以导出为纯推理模型：

```batch
python PaddleOCR\tools\export_model.py ^
    -c PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml ^
    -o Global.checkpoints=output\PP-OCRv4_mobile_det\iter_epoch_50 ^
    -o Global.save_inference_dir=.\inference\det
```

导出后，`inference\det\` 目录下会生成三个文件（`.pdmodel`、`.pdiparams`、`.pdiparams.info`），可以脱离 PaddleOCR 训练框架直接使用。

### Q9: conda 环境操作常见错误

**"conda: command not found" 或 "'conda' 不是内部或外部命令"**

说明你没有用 **Anaconda Prompt**。请在开始菜单搜索 "Anaconda Prompt" 并用它打开终端。不要使用 PowerShell 或普通的 CMD。

**环境创建失败，提示文件被占用**

以管理员身份运行 Anaconda Prompt，然后重试。或者手动删除报错提示的文件后重试。

---

## 命令速查

以下是你在这个项目中可能会反复用到的命令，按需查阅：

| 场景 | 命令 |
|------|------|
| 切换到项目目录 | `cd /d C:\Users\27695\Desktop\paddleocr_pjct` |
| 激活环境 | `conda activate mchlrn` |
| 安装/更新依赖 | `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 环境检查 | `python scripts\init_check.py` |
| 生成数据 | `python scripts\generate_data.py` |
| 启动训练（脚本） | `scripts\train_launcher.bat` |
| 启动训练（命令） | `python PaddleOCR\tools\train.py -c PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml` |
| 推理测试 | `python PaddleOCR\tools\infer_det.py -c PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml -o Global.infer_img=data\val\id_card_0000.jpg -o Global.checkpoints=output\PP-OCRv4_mobile_det\iter_epoch_50` |
| 修复 cuDNN DLL 路径 | `set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"` |

---

## 参考文献

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle 安装指南](https://www.paddlepaddle.org.cn/install/quick)
- [PP-OCRv4 论文](https://arxiv.org/abs/2305.13168)
- [DB (Differentiable Binarization) 论文](https://arxiv.org/abs/1911.08947)

## License

本项目仅供学习研究使用，请勿用于非法用途。
