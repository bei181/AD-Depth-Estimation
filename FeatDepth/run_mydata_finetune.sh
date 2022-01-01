## 第二阶段
## 多GPU训练

CUDA_VISIBLE_DEVICES=2,3 `which python` -m torch.distributed.launch --master_port=9901 \
--nproc_per_node=2 train.py --config config/cfg_my_fm_finetune.py --work_dir log/my_fm_finetune \