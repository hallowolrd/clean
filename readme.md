tmux new -s train_clean1
tmux attach -t train_clean1

cd /home/cjq/Project/clean
conda activate fl_moe

CUDA_VISIBLE_DEVICES=0 python a/uniform.py \
  --dataset cifar10 \
  --backbone resnet_cifar \
  --output-dir outputs/cifar10_resnet18

python plot_compare_acc.py \
  --input-dir outputs/cifar10_resnet_cifar \
  --window 5

支持的数据集：
cifar10
cifar100
cinic10
fashionmnist
stl10
tiny-imagenet-200
femnist

支持的backbone：
resnet_cifar
mobilenet_v2
vit_tiny