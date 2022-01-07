CUDA_VISIBLE_DEVICES=0 python -m depth_motion_field_train \
  --model_dir=log/kitti_experiment_test \
  --param_overrides='{
    "model": {
      "input": {
        "data_path": "/home/tata/Project/dataset/kitti_data_processed/train.txt"
      }
    },
    "trainer": {
      "init_ckpt": "/home/tata/Project/models/resnet-18-tensorflow/init/model.ckpt",
      "init_ckpt_type": "imagenet",
      "learning_rate": 1e-4
    }
  }'
