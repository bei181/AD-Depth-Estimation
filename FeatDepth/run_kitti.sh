## 第一阶段
# # 单GPU训练
# CUDA_VISIBLE_DEVICES=0 `which python` -m torch.distributed.launch train.py --config config/cfg_kitti_autoencoder.py --work_dir log/kitti_autoencoder

## 多GPU训练
# CUDA_VISIBLE_DEVICES=0,1 `which python` -m torch.distributed.launch --master_port=9900 \
# --nproc_per_node=2 train.py --config config/cfg_kitti_autoencoder.py --work_dir log/kitti_autoencoder \

## 第二阶段
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 `which python` -m torch.distributed.launch train.py --config config/cfg_kitti_fm.py --work_dir log/kitti_fmdepth

## 多GPU训练
# CUDA_VISIBLE_DEVICES=0,1 `which python` -m torch.distributed.launch --master_port=9900 \
# --nproc_per_node=2 train.py --config config/cfg_kitti_fm.py --work_dir log/kitti_fmdepth \

## Online Refinement (单GPU)
# CUDA_VISIBLE_DEVICES=0 `which python` -m torch.distributed.launch train.py --config config/cfg_kitti_fm_refine.py --work_dir log/kitti_fmdepth_refine

