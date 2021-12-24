import os
import numpy as np

# image_dir = '/home/swei/ztworks/ailab/python/data/castle-P30/images/'
image_dir = "/home/swei/ztworks/ailab/python/data/Herz-Jesus-P8/images/"
MRT = 0.7
#相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距，而K[0][2]和K[1][2]
#代表图像的中心像素。
# K = np.array([
#         [2362.12, 0, 720],
#         [0, 2362.12,  578],
#         [0, 0, 1]])
# K = np.array([
#         [361.54125, 0.0, 82.9005 ],
#         [0.0, 360.3975, 66.383625],
#         [0.0, 0.0, 1.0 ]])

K = np.array([
        [2759.48, 0.0, 1520.69],
        [0.0, 2764.16, 1006.81],
        [0.0, 0.0, 1.0 ]])

#选择性删除所选点的范围。
x = 0.5
y = 1