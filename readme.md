export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=3

tmux new -s train_clean1
tmux attach -t train_clean1

cd /home/cjq/Project/clean
conda activate fl_moe

CUDA_VISIBLE_DEVICES=1 python train.py --config configs/uniform.yaml
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/sample_weighted.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/fisher_only.yaml