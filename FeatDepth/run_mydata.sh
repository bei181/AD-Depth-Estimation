## 第一阶段
## 单GPU训练
# CUDA_VISIBLE_DEVICES=0 `which python` -m torch.distributed.launch train.py --config config/cfg_my_autoencoder.py --work_dir log/my_autoencoder

## 多GPU训练
# CUDA_VISIBLE_DEVICES=0,1 `which python` -m torch.distributed.launch --master_port=9900 \
# --nproc_per_node=2 train.py --config config/cfg_my_autoencoder.py --work_dir log/my_autoencoder \

## 第二阶段
# 单GPU训练
# CUDA_VISIBLE_DEVICES=0 `which python` -m torch.distributed.launch train.py --config config/cfg_my_fm.py --work_dir log/my_fmdepth

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1 `which python` -m torch.distributed.launch --master_port=9900 \
--nproc_per_node=2 train.py --config config/cfg_my_fm.py --work_dir log/my_fmdepth \