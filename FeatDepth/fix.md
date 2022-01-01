１) 在FeatDepth-master中创建dataset文件夹，并将kitti-data软链接到其中：
ln -s kitti_data_real_path ./datasets/

kitti数据集结构如下：
dataset/kitti_data
├── 2011_09_26
│   ├── 2011_09_26_drive_0001_sync
│   │   ├── image_00
│   │   ├── image_01
│   │   ├── image_02
│   │   ├── image_03
│   │   ├── oxts
│   │   └── velodyne_points
│   ├── 2011_09_26_drive_0002_sync

2) 下载resnet权重
可以从以下官方链接下载 resnet 预训练权重：
  | 'resnet18': ' https://download.pytorch.org/models/resnet18-5c106cde.pth ',
  | 'resnet34': ' https://download.pytorch.org/models/resnet34-333f7ec4.pth ',
  | 'resnet50': ' https://download.pytorch.org/models/resnet50-19c8e357.pth ',
  | 'resnet101': ' https://download.pytorch.org/models/resnet101-5d3b4d8f.pth ',
  | 'resnet152': ' https://download.pytorch.org/models/resnet152-b121ed2d.pth ',
  | 'resnext50_32x4d': ' https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth ',
  | 'resnext101_32x8d'：'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth ',
  | 'wide_resnet50_2': ' https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth ',
  | 'wide_resnet101_2'：' https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth '，