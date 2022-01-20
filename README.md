# AD-Depth-Estimation

## 代码

1. traditional depth estimation 包含sfm和sfs两种传统方法，对应深度估计课程实战课中的传统实战方法

2. 基于连续帧的自监督学习方法: monodepth2 和 FeatDepth

3. 处理动态物体的自监督学习方法: depth and motion learning 和 struct2depth

4. depth_estimate_application：是深度估计应用的一些代码，包括尺度恢复，伪点云生成，以及真实尺度下RGBD可视化


## 数据集

1. sfm所用数据集存放在百度网盘: https://pan.baidu.com/s/1bQpqF4Zn6O4C1WNfKNjcmQ 提取码: z682； 

2. sfs数据集存放在sfs代码中的res文件夹中

3. kitti数据集可以从官网下载: http://www.cvlibs.net/datasets/kitti/raw_data.php

也可以从百度网盘下载: https://pan.baidu.com/s/1qdoGlaRq8q-d0_bsAbrvyg 提取码: lu95 

4. cityscapes数据集官网下载链接: https://www.cityscapes-dataset.com/

5. 私有数据集存放在百度网盘: https://pan.baidu.com/s/1MguCs7r-l5Jbzm2gOhctUw 提取码: 735r 



## 预训练模型

1. depth_and_motion 在 kitti 和 cityscapes 上训练的模型

2. tensorflow 格式的 resnet 18 预训练权重

3. struct2depth 在 kitti 和 cityscapes 上训练的模型

都存放在百度网盘: https://pan.baidu.com/s/18oFEx5u8x_qwL7xXHAJyhg 提取码: 7kto

