CUDA_VISIBLE_DEVICES=0 python -m depth_motion_field_train \
  --model_dir=log/mydata_experiment \
  --param_overrides='{
    "model": 
      { "batch_size": 6,
        "input": 
        {
        "data_path": "/home/tata/Project/dataset/my_dataset/train_files.txt",
        "reader": 1
        },
        "image_preprocessing": {"data_augmentation": False,"image_height": 416,"image_width": 736}
      },
    "trainer": 
    {
      "learning_rate": 1e-5,
      "init_ckpt": "/home/tata/Project/models/resnet-18-tensorflow/init/model.ckpt",
      "init_ckpt_type": "imagenet"
    }
    }'
