import cv2
import os
import sys
import numpy as np

sys.path.append('./')
from util_project import get_pcd, lidar_to_pixel, draw_points, CameraIntrinsics_scaled, Lidar2CamMatrix


def statictics(d_list):
    d_arr = np.array(d_list)
    d_aver = np.mean(d_arr)
    d_med = np.median(d_arr)
    d_min = np.min(d_arr)
    d_max = np.max(d_arr)
    return d_aver, d_med, d_min, d_max


def to_3d_point(depth_map):
    points = np.ones((416, 736, 3))

    x = np.linspace(0,735,736)
    y = np.linspace(0,415,416)
    X, Y = np.meshgrid(x, y)

    points[:,:,0] = X 
    points[:,:,1] = Y 
    points[:,:,2] = depth_map # np.squeeze(depth_map,axis=2)
    points = points.reshape(-1,3)

    return points


def project_to_3d(depth_recover, cx, cy, fx, fy,width, height, truncate=False):
    # points = np.ones((1080, 1920, 4))
    points = np.ones((height, width, 3))
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    X = (X-cx)/fx
    Y = (Y-cy)/fy
    # depth_recover = np.squeeze(depth_recover,axis=2)
    depth_recover_mul = np.squeeze(np.stack((depth_recover,depth_recover,depth_recover),axis=2))
    points[:,:,0] = X 
    points[:,:,1] = Y 
    points[:,:,:3] = np.multiply(points[:,:,:3], depth_recover_mul)

    ## BGR 转 uint格式的rgb :RGB = R + G * 256 + B * 256 * 256
    # points[:,:,3] = img[:,:,2] + img[:,:,1]*256 + img[:,:,0] * 256 * 256
    # points = points.reshape((-1,4))
    points = points.reshape(-1,3)

    if truncate:
        new_points = list()
        # points[points[:,2] > truncate, 2] = -1 # 令Z为0
        for ii in range(points.shape[0]):
            if points[ii,2] < truncate:
                new_points.append(points[ii,:])
        points = np.array(new_points)


    return points

def points2pcd(points, PCD_PATH):
    # 写文件句柄
    handle = open(PCD_PATH, 'a')
    
    # 得到点云点数
    point_num=points.shape[0]

    # pcd头部（重要）
    # handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z \nSIZE 4 4 4 \nTYPE F F F \nCOUNT 1 1 1')

    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        # string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(points[i, 3])
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()

def get_scale_factor(depth_map,points):
    _, lidar_dpoints = lidar_to_pixel(points,  CameraIntrinsics_scaled, Lidar2CamMatrix, width=736, height=416)
    # 选定ROI区域
    x_range =(0,736)  # (420,1785) 
    y_range =  (300,416)  # (585,1080)
    # 计算ROI区域内的 Lidar点 和 深度图 的统计特征：均值，中值，最大值，最小值
    # 其中，深度图上只选取有Lidar点的像素点
    l_depth, c_depth = list(),list()
    for l_point in lidar_dpoints:
        if l_point[0] >= x_range[0] and l_point[0] <= x_range[1] and l_point[1] >= y_range[0] and l_point[1] <= y_range[1]:
            l_depth.append(l_point[2])
            c_depth.append(depth_map[l_point[1],l_point[0]])
    l_aver, l_med, l_min, l_max = statictics(l_depth)
    c_aver, c_med, c_min, c_max = statictics(c_depth)
    scale_factor =  l_med / c_med  
    return scale_factor

def scale_caculate(depth_dir, time_align_txt, img_dir, pcd_dir, out_dir, scale_txt):
    
    if os.path.exists(scale_txt):
        os.remove(scale_txt)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, "recovery")):
        os.mkdir(os.path.join(out_dir, "recovery"))
    if not os.path.exists(os.path.join(out_dir, "scale_recovery")):
        os.mkdir(os.path.join(out_dir, "scale_recovery"))
    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()

    with open(scale_txt,'x') as f_write:
        for ii, path in enumerate(align_paths):
            try:
                pcd_path = os.path.join(pcd_dir, path.split()[0])
                img_path = os.path.join(img_dir, path.split()[1])
                print(img_path, pcd_path)
                img_name = path.split()[1].split('.')[0]
                depth_path = os.path.join(depth_dir, img_name + '.npy')
                if not os.path.exists(depth_path):
                    print(depth_path, " is not exist")
                    continue
                depth_map = np.load(depth_path)  # 736*416
                
                points = get_pcd(pcd_path)
                img = cv2.imread(img_path)
                img = cv2.resize(img,(736,416))

                # depth_ = cv2.imread(depth_path,0)
                # depth_map = cv2.resize(depth_,(1920,1080))
                lidar_points, lidar_dpoints = lidar_to_pixel(points)
                # img_visual = draw_points(lidar_points, img)
                # print('saving visualization pictures.')
                # cv2.imwrite(os.path.join(out_dir, img_name + '_lidar_on_depth_map.png'), img_visual)
                # 选定ROI区域
                x_range = (0,736)  # (420,1785)
                y_range = (300,416)  # (585,1080)
                
                # 计算ROI区域内的 Lidar点 和 深度图 的统计特征：均值，中值，最大值，最小值
                # 其中，深度图上只选取有Lidar点的像素点
                l_depth, c_depth = list(),list()
                for l_point in lidar_dpoints:
                    if l_point[0] >= x_range[0] and l_point[0] <= x_range[1] and l_point[1] >= y_range[0] and l_point[1] <= y_range[1]:
                        l_depth.append(l_point[2])
                        c_depth.append(depth_map[l_point[1],l_point[0]])
                l_aver, l_med, l_min, l_max = statictics(l_depth)
                c_aver, c_med, c_min, c_max = statictics(c_depth)
                # print('lidar statictics: '+ str(l_aver)+', '+ str(l_med)+', '+ str(l_min)+', '+ str(l_max))
                # print('lidar statictics: '+ str(c_aver)+', '+ str(c_med)+', '+ str(c_min)+', '+ str(c_max))
                # 利用中值计算尺度因子
                scale_factor =  l_med / c_med  
                print('scale_factor: '+ str(scale_factor))
                f_write.write(img_name + ' ' + str(scale_factor) + '\n')

                # 尺度恢复
                depth_recover = depth_map * scale_factor
                # 保存尺度恢复后的深度图结果
                depth_rec_path = os.path.join(out_dir,'recovery', img_name + '_rec_depth.png')
                cv2.imwrite(depth_rec_path, depth_recover)

                # 2D像素点反投影至3D点
                cx,cy,fx,fy = 851.8287, 424.0041, 1998.7356, 1991.8909
                width, height = 736, 416
                points_3d = project_to_3d(depth_recover, cx,cy,fx,fy, width, height)
                print(points_3d)
                # 存成pcd格式
                PCD_PATH = os.path.join(out_dir,'scale_recovery', img_name + '_scale_recovery.pcd')
                points2pcd(points, PCD_PATH)
            except:
                print(ii)


def align_lidar_camera(time_align_txt, img_dir, pcd_dir):
    img_list = sorted(os.listdir(img_dir))
    pcd_list = sorted(os.listdir(pcd_dir))

    img_time_list = list()
    for img in img_list:
        time_stamp = float(img.split('.')[0])/1000  # ms
        img_time_list.append(time_stamp)

    if os.path.exists(time_align_txt):
        os.remove(time_align_txt)

    with open(time_align_txt,'x') as f_write:
        for ii, pcd in enumerate(pcd_list): # 对每一帧激光去找最近的图像
            time_stamp = float(pcd.split('.pcd')[0]) * 1000  # ms
            start = max(ii*2 - 100, 0)
            end = min(ii*2 + 50, len(img_time_list))
            img_search_list = img_time_list[start: end]   # 前5秒，后2.5秒
            diff_list = list(np.abs(np.array(img_search_list) - time_stamp))
            # print(min(diff_list))
            ind = diff_list.index(min(diff_list))
            try:
                select_img = img_list[start+ind-1]  # 找最近时间戳的前一帧
            except: 
                select_img = img_list[start+ind]
            # print(select_img)
            f_write.write(pcd+' '+select_img+'\n')


def main():
    depth_dir = '../dataset/hs5_0831/depth_map/hs5_35k_3_east_park'
    out_dir = 'output/vis_0831_east_park'
    scale_txt = os.path.join(out_dir, 'scale_factor_east_park.txt')
    time_align_txt = '../dataset/hs5_0831/lidar_camera_align.txt'
    img_dir = '../dataset/hs5_0831/image_undistort'
    pcd_dir = '../dataset/hs5_0831/pcl_pcd'

    align_lidar_camera(time_align_txt, img_dir, pcd_dir)  # 对齐激光雷达和相机
    scale_caculate(depth_dir, time_align_txt, img_dir, pcd_dir, out_dir, scale_txt)
    

if __name__ == '__main__':
    main()