#!/bin/bash
# run_two_stage.sh
# 两阶段训练启动脚本（双卡 DDP）
#
# 用法:
#   bash run_two_stage.sh both      # 跑两阶段全程（默认）
#   bash run_two_stage.sh stage1    # 只跑预训练
#   bash run_two_stage.sh stage2    # 只跑 SFT（需先有 stage1）
#
# 环境变量:
#   GPU_COUNT=2              # 用几张卡，默认 2
#   CUDA_VISIBLE_DEVICES=5,6 # 可选: 指定具体 GPU，需放在 bash 命令前

set -e
cd /apps/users/xzl/mini_LLaMA
PY=/apps/users/xzl/miniconda3/envs/qwen3vl/bin/python
LOG_DIR=/apps/users/xzl/mini_LLaMA/logs
mkdir -p "$LOG_DIR"

GPU_COUNT=${GPU_COUNT:-2}
STAGE="${1:-both}"

run_stage1() {
    echo "==========================================="
    echo "  Stage 1: 预训练 (从头, ${GPU_COUNT} 卡 DDP)"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "  $(date)"
    echo "==========================================="
    torchrun --standalone --nproc_per_node=$GPU_COUNT \
        /apps/users/xzl/mini_LLaMA/scripts/05a_pretrain.py \
        2>&1 | tee "$LOG_DIR/stage1.log"
}

run_stage2() {
    echo "==========================================="
    echo "  Stage 2: SFT (基于 Stage 1, ${GPU_COUNT} 卡 DDP)"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "  $(date)"
    echo "==========================================="
    if [ ! -d "/apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage1/final" ]; then
        echo "❌ 错误: 找不到 Stage 1 模型"
        echo "   /apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage1/final"
        echo "   请先跑 stage1"
        exit 1
    fi
    torchrun --standalone --nproc_per_node=$GPU_COUNT \
        /apps/users/xzl/mini_LLaMA/scripts/05b_sft.py \
        2>&1 | tee "$LOG_DIR/stage2.log"
}

case "$STAGE" in
    stage1)
        run_stage1
        ;;
    stage2)
        run_stage2
        ;;
    both|*)
        run_stage1
        run_stage2
        echo ""
        echo "✅ 两阶段训练完成!"
        echo "   预训练模型: /apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage1/final"
        echo "   SFT 模型:   /apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage2/final"
        echo ""
        echo "下一步推理:"
        echo "  $PY /apps/users/xzl/mini_LLaMA/scripts/04_infer.py \\"
        echo "    --model_dir /apps/users/xzl/mini_LLaMA/checkpoints/two_stage/stage2/final"
        ;;
esac
