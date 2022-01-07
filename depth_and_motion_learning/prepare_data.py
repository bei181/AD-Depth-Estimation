import os
from glob import glob
import fnmatch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def recursiveSearchFiles(dirPath, partFileInfo): 
    fileList = []
    pathList = glob(os.path.join(dirPath, '*'))#windows path
    for mPath in pathList:
        if fnmatch.fnmatch(mPath, partFileInfo):
            fileList.append(mPath) #符合条件条件加到列表
        elif os.path.isdir(mPath):
            fileList += recursiveSearchFiles(mPath, partFileInfo) #将返回的符合文件列表追加到上层
        else:
            pass
    return fileList


def generate_kitti_list():
    root_path = './dataset/kitti_data_processed'
    output_txt = os.path.join(root_path,'train.txt')
    file_list = recursiveSearchFiles(root_path,'*.png')
    if os.path.exists(output_txt):
        os.remove(output_txt)

    with open(output_txt,'x') as f_write:
        for file in file_list:
            img_path = file.split('/')
            folder = img_path[3]
            img_name = img_path[4].split('.')[0]
            f_write.write(folder+' '+ img_name+'\n')


def generate_kitti_test_list():
    root_path = '/home/tata/Project/dataset/kitti_data/2011_09_26/2011_09_26_drive_0019_sync/image_02/data'
    output_txt = os.path.join('./dataset','kitti_test.txt')
    file_list = recursiveSearchFiles(root_path,'*.jpg')
    if os.path.exists(output_txt):
        os.remove(output_txt)

    file_list = sorted(file_list)
    num_frames = len(file_list)
    with open(output_txt,'x') as f_write:
        for id in range(num_frames):
            file_name = file_list[id].split('/')[-1]
            f_write.write('kitti_data/2011_09_26/2011_09_26_drive_0019_sync/image_02/data/'+file_name+'\n')
            

def generate_my_list():
    root_dir = '/home/tata/Project/dataset/my_dataset/'
    img_path = os.path.join(root_dir,'image_undistort')
    data_txt = os.path.join(root_dir,'train_files.txt')
    if os.path.exists(data_txt):
        os.remove(data_txt)

    all_frames = sorted(os.listdir(img_path))
    num_frames = len(all_frames)

    with open(data_txt,'x') as f_write:
        for id in range(num_frames):
            file_name = all_frames[id]
            img_path = os.path.join('image_undistort',file_name)
            if id == 0:
                f_write.write(img_path+' start\n')
            else:
                f_write.write(img_path+'\n')


def generate_compare_video():
    root_path = 'depth_and_motion/results'
    video_name = 'depth_my.avi'
    img_path = os.path.join(root_path,'img')
    video_path = os.path.join(root_path,video_name)
    file_list = sorted(os.listdir(img_path))
    n_sample = int(len(file_list)/2)
 
    fps = 5
    width,height = 576, 320
    size = (width, int(height*2))
    videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    for i in range(n_sample):
        print(str(i)+'/'+ str(float(n_sample)))
        ind = int(i*2)
        depth_array = cv2.imread(os.path.join(img_path, file_list[ind]))
        img_array = cv2.imread(os.path.join(img_path, file_list[ind+1]))
        img_array = cv2.resize(img_array,(width,height))
        depth_array = cv2.resize(depth_array,(width,height))
        fusion_img = np.vstack((img_array, depth_array))

        videowriter.write(fusion_img)


def main():
    # generate_compare_video()
    # generate_kitti_list()
    # generate_my_list()
    generate_kitti_test_list()


if __name__ =='__main__':
    main()