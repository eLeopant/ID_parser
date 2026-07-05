#!/bin/bash
# ============================================================
# PaddleOCR 检测模型微调启动脚本
# 使用 PP-OCRv4 mobile 检测模型 (student)
# ============================================================

# 获取脚本所在目录的上级目录（项目根目录）
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# 设置 cuDNN 库路径（conda 安装的 cuDNN）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 配置文件路径
CONFIG_PATH="./PaddleOCR/configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml"

# 训练命令
echo "=" * 60
echo "  开始训练 PP-OCRv4 mobile 检测模型"
echo "  配置文件: $CONFIG_PATH"
echo "  训练数据: ./data/train/"
echo "  验证数据: ./data/val/"
echo "  Epoch: 50, Batch Size: 8"
echo "=" * 60

python ./PaddleOCR/tools/train.py \
    -c "$CONFIG_PATH"

echo ""
echo "训练完成！模型保存在: ./output/PP-OCRv4_mobile_det/"
