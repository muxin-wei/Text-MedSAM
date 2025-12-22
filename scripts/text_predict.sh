#!/bin/bash

# ================= 配置区域 =================
# 项目根目录
PROJECT_ROOT="/root/autodl-tmp/Rep-MedSAM"

# 推理脚本路径
SCRIPT_PATH="${PROJECT_ROOT}/scripts/batch_infer_text.py"

# Checkpoint 所在目录
CKPT_DIR="${PROJECT_ROOT}/ckpts"

# 数据集和输出路径 (根据你之前的代码补充)
IMG_DIR="/root/autodl-tmp/dataset/val"
GT_DIR="/root/autodl-tmp/dataset/val_gt"
OUTPUT_DIR="/root/autodl-tmp/dataset/seg"

# ===========================================

# 切换到项目根目录，防止路径引用问题
cd "$PROJECT_ROOT" || { echo "无法切换到目录 $PROJECT_ROOT"; exit 1; }

echo "开始批量推理任务..."
echo "Checkpoint 目录: $CKPT_DIR"

# 查找所有 .ckpt 文件，排序后循环执行
# 使用 find + sort 确保按 epoch/step 顺序执行
find "$CKPT_DIR" -name "*.ckpt" | sort | while read -r ckpt_path; do
    
    ckpt_filename=$(basename "$ckpt_path")
    echo "---------------------------------------------------------------"
    echo "正在处理 Checkpoint: $ckpt_filename"
    echo "完整路径: $ckpt_path"
    echo "---------------------------------------------------------------"

    # 执行推理脚本
    # 注意：这里补全了 img_dir, gt_dir 等必要参数，如不需要可删除
    python "$SCRIPT_PATH" \
        --checkpoint_path "$ckpt_path" \
        --img_dir "$IMG_DIR" \
        --gt_dir "$GT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --image-size 256 \
        --ds-scale 4.0

    # 检查上一条命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ Checkpoint $ckpt_filename 推理完成。"
    else
        echo "❌ Checkpoint $ckpt_filename 推理失败！"
        # 如果希望遇到错误立即停止整个脚本，请取消下面这行的注释
        # exit 1
    fi

    echo ""
done

echo "所有任务执行完毕。"