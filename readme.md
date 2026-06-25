tmux new -s train_clean1
tmux attach -t train_clean1

cd /home/cjq/Project/clean
conda activate fl_moe

CUDA_VISIBLE_DEVICES=2 python train.py --config configs/uniform.yaml
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/sample_weighted.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/fisher_kfac_expert.yaml

python tools/plot_compare_test_acc.py \
  --runs \
  uniform=outputs/cifar10_c20_a0p1_resnet_sparse_moe_head_e4_top1_r200_ep5_neuniform_exuniform_s0/results.csv \
  kfac=outputs/cifar10_c20_a0p1_resnet_sparse_moe_head_e4_top1_r200_ep5_neuniform_exfisher_kfac_expert_s0/results.csv \
  --window 5 \
  --hide-raw \
  --out outputs/c20_200x5.png