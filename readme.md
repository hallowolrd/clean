tmux new -s train_clean1
tmux attach -t train_clean1

cd /home/cjq/Project/clean
conda activate fl_moe

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/uniform.yaml
CUDA_VISIBLE_DEVICES=2 python train.py --config configs/sample_weighted.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/fisher_kfac_expert.yaml

python tools/plot_compare_test_acc.py \
  --runs \
  uniform=outputs_bias/cifar10_c5_a0p1_resnet_sparse_moe_head_e4_top2_r100_ep5_neuniform_exuniform_s0/results.csv \
  kfac=outputs_bias/cifar10_c5_a0p1_resnet_sparse_moe_head_e4_top2_r100_ep5_neuniform_exfisher_kfac_expert_s0/results.csv \
  --window 1 \
  --hide-raw \
  --out pictures/include_bias_50_1.png