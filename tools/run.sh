#!/usr/bin/env bash
set -u

# ============================================================
# CIFAR10 + ResNet18-like (resnet_cifar)
# 5 个专家聚合实验并行启动（不包含 pFedMoE）
# ============================================================

# 项目根目录：默认在 ~/Project/clean 里执行本脚本。
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Project/clean}"
SCRIPT_DIR="$PROJECT_ROOT/a"

DATASET="cifar10"
BACKBONE="resnet_cifar"
OUTPUT_DIR="$PROJECT_ROOT/outputs/${DATASET}_${BACKBONE}"
LAUNCH_LOG_DIR="$OUTPUT_DIR/launcher_logs"

# ------------------------------------------------------------
# GPU 分配
# 直接修改下面 5 个数字即可。
# 也可以在启动脚本时用环境变量临时覆盖。
# ------------------------------------------------------------
GPU_UNIFORM="${GPU_UNIFORM:-0}"
GPU_FISHER="${GPU_FISHER:-0}"
GPU_FED_MOE="${GPU_FED_MOE:-0}"
GPU_FEDMOE_DA="${GPU_FEDMOE_DA:-1}"
GPU_SOMFED="${GPU_SOMFED:-1}"

mkdir -p "$LAUNCH_LOG_DIR"

# 检查脚本文件是否存在，避免路径写错后静默失败。
for file in \
    uniform.py \
    fisher_kfac_expert.py \
    fed_moe_style_expert.py \
    fedmoe_da_style_expert.py \
    somfed_style_expert.py
do
    if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
        echo "[ERROR] Missing: $SCRIPT_DIR/$file" >&2
        exit 1
    fi
done

launch_experiment() {
    local name="$1"
    local gpu="$2"
    local script="$3"
    local log_file="$LAUNCH_LOG_DIR/${name}.log"
    local pid_file="$LAUNCH_LOG_DIR/${name}.pid"

    echo "[Launch] $name | GPU=$gpu | script=$script"

    nohup env CUDA_VISIBLE_DEVICES="$gpu" \
        python "$SCRIPT_DIR/$script" \
        --dataset "$DATASET" \
        --backbone "$BACKBONE" \
        --output-dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    echo "         PID=$pid"
    echo "         log=$log_file"
}

launch_experiment "uniform"   "$GPU_UNIFORM"   "uniform.py"
launch_experiment "fisher"    "$GPU_FISHER"    "fisher_kfac_expert.py"
launch_experiment "fed_moe"   "$GPU_FED_MOE"   "fed_moe_style_expert.py"
launch_experiment "fedmoe_da" "$GPU_FEDMOE_DA" "fedmoe_da_style_expert.py"
launch_experiment "somfed"    "$GPU_SOMFED"    "somfed_style_expert.py"

echo
echo "============================================================"
echo "All 5 experiments have been launched in parallel."
echo "Dataset : $DATASET"
echo "Backbone: $BACKBONE"
echo "Output  : $OUTPUT_DIR"
echo "Logs    : $LAUNCH_LOG_DIR"
echo "============================================================"
echo
echo "GPU mapping:"
echo "  uniform   -> GPU $GPU_UNIFORM"
echo "  fisher    -> GPU $GPU_FISHER"
echo "  fed_moe   -> GPU $GPU_FED_MOE"
echo "  fedmoe_da -> GPU $GPU_FEDMOE_DA"
echo "  somfed    -> GPU $GPU_SOMFED"
echo
echo "Check processes:"
echo "  ps -fp \$(cat $LAUNCH_LOG_DIR/*.pid 2>/dev/null)"
echo
echo "Check GPU usage:"
echo "  nvidia-smi"
