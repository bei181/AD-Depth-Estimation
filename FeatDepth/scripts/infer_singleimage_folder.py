from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset
from mono.datasets.folder_dataset import FolderDataset 

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)



def evaluate(cfg_path,model_path,gt_path, output_path):
    cfg = Config.fromfile(cfg_path)

    dataset = FolderDataset(cfg['in_path'],
                              cfg['height'],
                              cfg['width'],
                              [0],
                              is_train=False,
                              img_ext='jpg',
                              gt_depth_path=gt_path)

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    cfg.model['imgs_per_gpu'] = 1
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            print(batch_idx)
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            outputs = model(inputs)

            img_path = os.path.join(output_path, 'img_{:0>4d}.jpg'.format(batch_idx))
            plt.imsave(img_path, inputs[("color", 0, 0)][0].squeeze().transpose(0,1).transpose(1,2).cpu().numpy())

            disp = outputs[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp[0, 0].cpu().numpy()
            pred_disp = cv2.resize(pred_disp, (cfg['width'], cfg['height']))

            img_path = os.path.join(output_path, 'disp_{:0>4d}.jpg'.format(batch_idx))
            vmax = np.percentile(pred_disp, 95)
            plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)

    print("\n-> Done!")


if __name__ == "__main__":
    cfg_path = './config/cfg_my_test.py' # path to cfg file
    model_path = './log/my_fmdepth_finetune/epoch_31.pth'# path to model weight
    gt_path = None  # path to kitti gt depth
    output_path = 'results/test_my_fm_finetune/' # dir for saving depth maps
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    evaluate(cfg_path,model_path,gt_path,output_path)