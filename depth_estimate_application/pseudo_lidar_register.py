import numpy as np
import os
import sys
import cv2
# import mayavi.mlab

from pyntcloud import PyntCloud
from pandas import DataFrame
import open3d as o3d

from scale_recovery import project_to_3d, get_scale_factor
from util_project import get_pcd, project_pc_2d, Lidar2CamMatrix, CameraIntrinsics_scaled
from clustering.cluster import remove_ground 
from ICP.icp_open3d import point_registration

raw_width, raw_height = 1920, 1080
scaled_width, scaled_height = 736,416
scale_x, scale_y = scaled_width/raw_width, scaled_height/raw_height
cx,cy,fx,fy = 851.8287*scale_x, 424.0041*scale_y, 1998.7356*scale_x, 1991.8909*scale_y


def get_rgbd_points(img_name, depth_scale, depth_dir, truncate):
    try:
        depth_scale = depth_scale.strip()
    except:
        pass
    depth_map = np.load(os.path.join(depth_dir, img_name+'.npy'))
    depth_map *= float(depth_scale)
    pseudo_points = project_to_3d(depth_map, cx, cy, fx, fy, scaled_width, scaled_height, truncate)

    pseudo_points_filtered = list()
    for point in pseudo_points:
        if point[2] > 0.5:
            pseudo_points_filtered.append(point)

    return np.array(pseudo_points_filtered)


def lidar2camera_coordinates(points,truncate=100):
    camera_coor_points = np.matmul(Lidar2CamMatrix, points.transpose())[:3,:] # shape:(3,N), 相机坐标系下的3d点
    camera_coor_points_norm = camera_coor_points / camera_coor_points[2,:]  # Normalization, shape:(3,N)
    pixel_points = np.matmul(CameraIntrinsics_scaled, camera_coor_points_norm) # shape:(2,N)
    pixel_points = pixel_points.transpose() # shape:(N,2)

    # 超出图像范围的点丢掉
    lidar_points_trunc = list()
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i,:]
        if x>0  and x < scaled_width and  y>0 and y < scaled_height and camera_coor_points[2,i] < truncate:
        # if camera_coor_points[2,i] < 55:  # 只截取离相机60米以内的点
            lidar_points_trunc.append(camera_coor_points[:,i].transpose())

    return np.array(lidar_points_trunc), camera_coor_points.transpose()  # 第二项是原始激光点


def lidar_in_img(lidar_points, Lidar2Cam, CameraInt, range_,truncate):  ## 筛选图像范围内的lidar点
    camera_coor_points = np.matmul(Lidar2Cam, lidar_points.transpose())[:3,:] # shape:(3,N), 相机坐标系下的3d点
    camera_coor_points_norm = camera_coor_points / camera_coor_points[2,:]  # Normalization, shape:(3,N)
    pixel_points = np.matmul(CameraInt, camera_coor_points_norm) # shape:(2,N)
    pixel_points = pixel_points.transpose() # shape:(N,2)

    # 超出图像范围的点丢掉
    lidar_points_filter = list()
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i,:]
        if x > range_[0]  and x < range_[1] and  y > range_[2] and y < range_[3] and lidar_points[i,0] < truncate:  # 截取一定范围内的点
            lidar_points_filter.append(lidar_points[i,:3])

    return np.array(lidar_points_filter)


def get_lidar_points(align_paths, img_name, lidar_dir, range_, truncate, to_cam_coor = False):
    for paths in align_paths:
        lidar, img = paths.split()
        img_ = img.split('.')[0]

        if img_ == img_name:
            lidar_path = os.path.join(lidar_dir, lidar)
            lidar_points = get_pcd(lidar_path)  # (x,y,z,inten) 

            if to_cam_coor:
                lidar_points_trunc, lidar_points = lidar2camera_coordinates(lidar_points,truncate=truncate) #(x,y,z)
                return lidar_points_trunc, lidar_points
            else:
                lidar_points_trunc = lidar_in_img(lidar_points, Lidar2CamMatrix, CameraIntrinsics_scaled, range_, truncate)
                return lidar_points_trunc


def vis_points_pair(pseudo_points, lidar_points):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    # lidar_points
    points = DataFrame(lidar_points[:, 0:3])  # 选取每一列的第0个元素到第二个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    point_cloud_o3d_lidar = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    point_cloud_o3d_lidar.paint_uniform_color([1, 0.706, 0])
    point_cloud_o3d_lidar.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # pseudo points
    points = DataFrame(pseudo_points[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    point_cloud_o3d_pseudo = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    point_cloud_o3d_pseudo.paint_uniform_color([0, 0.651, 0.929])
    point_cloud_o3d_pseudo.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # # pseudo points create from rgbd image
    # color_raw = o3d.io.read_image("color.jpg")
    # depth_raw = o3d.io.read_image("depth.png") 
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(width=img_width, height=img_height, fx =fx, fy=fy, cx=cx, cy=cy)
    # rgbd_img = o3d.geometry.RGBDImage()
    # rgbd_img.create_from_color_and_depth(color_raw, depth_raw, depth_trunc=70.0, convert_rgb_to_intensity=False)
    # pcd = o3d.geometry.PointCloud()
    # pcd.create_from_rgbd_image(rgbd_img, intrinsic)


    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([point_cloud_o3d_lidar,point_cloud_o3d_pseudo], key_to_callback)
    # o3d.visualization.draw_geometries([point_cloud_o3d_lidar,point_cloud_o3d_pseudo])

 ## mayavi visualization method
    # x = lidar_points[:,0]
    # y = lidar_points[:,1]
    # z = lidar_points[:,2]
    # fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    # mayavi.mlab.points3d(x, y, z,
    #                     # s,  # Values used for Color
    #                     mode="point",
    #                     # colormap='spectral',  # 'bone', 'copper', 'gnuplot'
    #                     color=(0, 1, 0),   # Used a fixed (r,g,b) instead
    #                     figure=fig)
    # mayavi.mlab.show()
 ##
    return True

def pseudo_lidar_registration(lidar_dir, depth_dir, img_dir, time_align_txt, thres=20, output_dir=None, lidar_to_cam = False, visual = False, match = True):


    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()

    for ii, path in enumerate(align_paths):
        print(path)
        pcd_name, img_name = path.split()
        img_name = img_name.split('.')[0]
        depth_map = np.load(os.path.join(depth_dir, img_name+'.npy'))
        pcd_path = os.path.join(lidar_dir, pcd_name)
        points = get_pcd(pcd_path)

        # 利用深度图和尺度因子，并转化到相机坐标系下的3d点
        depth_scale = get_scale_factor(depth_map, points)
        print(depth_scale)
        pseudo_points = get_rgbd_points(img_name, depth_scale, depth_dir, truncate = thres)

        # 读激光data，并转化到相机坐标系下的3d点
        range_ = [0,736,0,416]
        lidar_points_truc, lidar_points = get_lidar_points(align_paths, img_name, lidar_dir, range_, truncate =thres, to_cam_coor = lidar_to_cam)

        if visual: # 可视化伪激光和激光点云
            vis_points_pair(pseudo_points, lidar_points_truc)

        if match:   # 激光与伪激光的配准
            # Use my optimization method to calibration
            # Point Cloud Registration with ICP
            lidar_points_trans, scale_p, transformation = point_registration(pseudo_points,lidar_points_truc, visual = True)
            img = cv2.imread(os.path.join(img_dir,img_name+'.jpg'))
            img = cv2.resize(img,(scaled_width, scaled_height))
            save_path = os.path.join(output_dir, img_name+'_icp_part.png')
            
            # Apply the transformation
            lidar_points_truc = np.hstack((lidar_points_truc, np.ones((lidar_points_truc.shape[0],1))))
            lidar_points_icp = np.matmul(transformation, lidar_points_truc.transpose())[:3,:]
            project_pc_2d(img, lidar_points_icp, save_path, CameraIntrinsics_scaled, scaled_width, scaled_height)


def remove_ground_points():
    # Remove Ground Points
    lidar_points = np.load('calib_data/lidar_points.npy').astype(np.float32)
    pseudo_points = np.load('calib_data/pseudo_lidar.npy').astype(np.float32)
    pseudo_points_filter = remove_ground(pseudo_points, method= 'RANSAC', visual=True)
    np.save('calib_data/pseudo_lidar_no_ground.npy', np.array(pseudo_points_filter.points))

    # Verify the Transform Matrix
    lidar_points = np.load('calib_data/lidar_points_trun15.npy').astype(np.float32)
    pseudo_points = np.load('calib_data/pseudo_lidar_trun15.npy').astype(np.float32)
    lidar2camera_matrix_icp = np.array([[ 0.3769053 , -0.8937396 , -0.24325311 ,-1.7898636],
                                        [0.00419791 , 0.26426598 ,-0.96444106 , 0.22657132],
                                        [0.92624223 , 0.36248147 , 0.10335464, -0.5009727],
                                        [ 0. ,         0. ,         0.   ,       1.        ]])
                                        
    lidar_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0],1))))
    camera_coor_points = np.matmul(lidar2camera_matrix_icp, lidar_points.transpose())[:3,:]
    vis_points_pair(pseudo_points, camera_coor_points.transpose())

def main():
    lidar_dir = '../dataset/hs5_0831/pcl_pcd'
    depth_dir = '../dataset/hs5_0831/depth_map/online_calib'
    img_dir = '../dataset/hs5_0831/image_undistort'
    time_align_txt = '../dataset/hs5_0831/online_calib.txt'
    # Compare lidar and pseudo_lidar
    truncate = 20
    output_dir = 'output/calib_data/project_lidar_0831_calib'
    visual = False
    pseudo_lidar_registration(lidar_dir, depth_dir, img_dir, time_align_txt, output_dir=output_dir, thres=truncate, lidar_to_cam = True, visual = visual)


if __name__ == '__main__':
    main()