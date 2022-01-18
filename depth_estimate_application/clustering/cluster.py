import os
import sys
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from visual.vis_o3d import vis_points, np2pcd, display_inlier_outlier
from util_project import get_pcd, Lidar2CamMatrix, CameraIntrinsics, label_mapping_nusc_2, map_label, paint_cluster_on_img


## Config of HS5
h_camera = 1.793
distance_thres = 1000

def point_in_img(points, range_):  ## 筛选图像范围内的lidar点
    camera_coor_points = np.matmul(Lidar2CamMatrix, points.transpose())[:3,:] # shape:(3,N), 相机坐标系下的3d点
    camera_coor_points_norm = camera_coor_points / camera_coor_points[2,:]  # Normalization, shape:(3,N)
    pixel_points = np.matmul(CameraIntrinsics, camera_coor_points_norm) # shape:(2,N)
    pixel_points = pixel_points.transpose() # shape:(N,2)

    # 超出图像范围的点丢掉
    lidar_points = list()
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i,:]
        if camera_coor_points[2,i] < distance_thres and  camera_coor_points[1,i] > h_camera - 5:  
            # 只截取离相机一定距离以及一定高度以下的点
            if x > range_[0]  and x < range_[1] and  y > range_[2] and y < range_[3]:  # 截取一定范围内的点
                lidar_points.append(camera_coor_points[:,i].transpose())

    return np.array(lidar_points)

def postprocess_cluster(clusters): # 希望分开连在一起的两个目标（一般是比较大的目标）
    for point_cloud in clusters:
        # 半径滤波
        pcd_filter, ind = point_cloud.remove_radius_outlier(nb_points=3, radius=1) 
        # 再聚类
        labels = np.array(pcd_filter.cluster_dbscan(eps=0.7, min_points=3, print_progress=False))
        max_label = labels.max()
        if max_label > 0: # 如果分成两个cluster了
            print(max_label)
    return clusters


def assign_grid(point, grid_coor):
    x,z = point[0],point[2]
    for ii, coordinate in enumerate(grid_coor):
        if x >= coordinate[0] and x <= coordinate[1]:
            if z >= coordinate[2] and x <= coordinate[3]:
                return ii


def concate_list_array(points):
    for index in range(len(points)):
        if index == 0:
            points_array = points[0]
        else:
            points_array = np.vstack((points_array, points[index]))
    return points_array


## Remove ground points 
def remove_ground_grid(pcd, visual = False):
    thres =2

    cam_points = np.asarray(pcd.points)
    n_x, n_z = (4,4)   # 3*3 个grid
    ## 确定bev的区间范围
    x_min, x_max = np.min(cam_points[:,0]), np.max(cam_points[:,0])
    z_min, z_max = np.min(cam_points[:,2]), np.max(cam_points[:,2])
    x_grid = np.linspace(x_min, x_max, n_x+1)
    y_grid = np.linspace(z_min, z_max, n_z+1)
    grid_coor = list()
    for x_id in range(x_grid.shape[0]-1):
        for y_id in range(y_grid.shape[0]-1):
            grid_coor.append((x_grid[x_id], x_grid[x_id+1], y_grid[y_id], y_grid[y_id+1]))

    ##  将点按照grid分组
    grid_index = [[] for i in range(n_x * n_z)] 
    for ii in range(cam_points.shape[0]):
        point = cam_points[ii,:]
        index = assign_grid(point, grid_coor)
        grid_index[index].append(point)

    ## 根据Grid筛选地面点
    points_no_ground, points_ground = [],[]
    for ii in range(n_x * n_z):
        if len(grid_index[ii]) < 1:
            continue
        points_set = np.array(grid_index[ii])
        y_bottom = np.max(points_set[:,1])  # 在最低点以上0.2米的点都视作地面点
        points_index = np.where(points_set[:,1] < (y_bottom-thres))
        points_filter = points_set[points_index]
        points_no_ground.append(points_filter)

        points_g = points_set[np.where(points_set[:,1] >= (y_bottom-thres))]
        points_ground.append(points_g)

    ground_cloud = np2pcd(concate_list_array(points_ground))
    obj_cloud = np2pcd(concate_list_array(points_no_ground))

    ## 可视化
    if visual:
        ground_cloud.paint_uniform_color([1.0, 0, 0])
        obj_cloud.paint_uniform_color([0, 0, 1.0])
        o3d.visualization.draw_geometries([obj_cloud, ground_cloud], zoom=0.8,
                                            front=[-0.4999, -0.1659, -0.8499],
                                            lookat=[2.1813, 2.0619, 2.0999],
                                            up=[0.1204, -0.9852, 0.1215])

    return obj_cloud


def remove_ground_ransac_half(pcd, visual= False): # 分成两半拟合模型
    raw_points = np.asarray(pcd.points)
    middle = np.mean(raw_points[:,0])
    index_left = list(np.where(raw_points[:,0] < middle+20)[0])
    index_right = list(np.where(raw_points[:,0] > middle-20)[0])
    pcd_left = pcd.select_by_index(index_left)
    pcd_right = pcd.select_by_index(index_right)

    plane_model_l, inliers_l = pcd_left.segment_plane(distance_threshold=0.15, ransac_n=100,  num_iterations=1000)
    outlier_cloud_l = pcd_left.select_by_index(inliers_l, invert=True)
    inlier_cloud_l = pcd_left.select_by_index(inliers_l)
    plane_model_r, inliers_r = pcd_right.segment_plane(distance_threshold=0.15, ransac_n=100,  num_iterations=1000)
    outlier_cloud_r = pcd_right.select_by_index(inliers_r, invert=True)
    inlier_cloud_r = pcd_right.select_by_index(inliers_r)

    points_no_ground = np.vstack((np.asarray(outlier_cloud_l.points), np.asarray(outlier_cloud_r.points)))
    points_no_ground = np.unique(points_no_ground, axis=0)

    if visual:
        inlier_cloud_l.paint_uniform_color([1.0, 0, 0])
        inlier_cloud_r.paint_uniform_color([1.0, 0, 0])

        o3d.visualization.draw_geometries([inlier_cloud_l, outlier_cloud_l, inlier_cloud_r, outlier_cloud_r], zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])


    return np2pcd(points_no_ground)


def remove_ground(points, method= 'RANSAC', visual=False):  ##  使用RANSAC拟合平面, 去除地面点
    pcd = np2pcd(points)

    # ##  统计滤波
    # # o3d.visualization.draw_geometries([pcd])
    # pcd_filter, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=3.0)
    # display_inlier_outlier(pcd, ind)

    # 半径滤波
    pcd_filter, ind = pcd.remove_radius_outlier(nb_points=5, radius=2)
    # display_inlier_outlier(pcd, ind)

    if method == 'RANSAC':  # Ransac 拟合平面
        plane_model, inliers = pcd_filter.segment_plane(distance_threshold=0.2, ransac_n=20,  num_iterations=1000)
        [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0") 
        outlier_cloud = pcd_filter.select_by_index(inliers, invert=True)
        if visual:
            inlier_cloud = pcd_filter.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=0.8,
                                            front=[-0.4999, -0.1659, -0.8499],
                                            lookat=[2.1813, 2.0619, 2.0999],
                                            up=[0.1204, -0.9852, 0.1215])

    elif method == 'RANSAC_HALF':
        outlier_cloud = remove_ground_ransac_half(pcd_filter, visual)
    
    elif method == 'GRID': # 栅格法
        outlier_cloud = remove_ground_grid(pcd_filter, visual)

    return outlier_cloud


def remove_ground_cyliner(raw_points, label):
    filter_points = list()
    for ii, point in enumerate(raw_points):
        if label[ii] == 1 and  point[0] > 0.01:
            filter_points.append(point)
    filter_points = np.array(filter_points)
    return filter_points


## Clustering and Postprocessing
def remove_non_object(labels, pcd, visual= False):
    bbox_vis_list = [pcd]
    bbox_select_list = []
    pcd_filter = []
    label_filter = []

    label_ind = 0
    for ii in range(labels.max()+1): # 对每一个目标，求它的bbox
        index = list(np.where(labels == ii)[0])
        obj_cloud = pcd.select_by_index(index)

        aabb = obj_cloud.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        bbox_vis_list.append(aabb)
        # obb = obj_cloud.get_oriented_bounding_box()
        # obb.color = (1, 0, 0)
        # bbox_list.append(obb)

        ## 过滤位置不合适的目标：空中的 & 贴在地上的
        y_bottom = aabb.max_bound[1]  # 底部的y坐标
        y_top = aabb.min_bound[1]  # 顶部的y坐标
        proper_loc = (y_bottom > 1.7 and y_top < 3)

        ## 过滤尺寸不合适的目标
        volum = aabb.volume()
        height = y_bottom - y_top
        length = aabb.max_bound[2] - aabb.min_bound[2]
        width = aabb.max_bound[0] - aabb.min_bound[0]
        proper_size = (height>0.8 and height<5 and length>0.3 and length<16 and width>0.3 and width<16 and volum>0.1 and volum<150)

        if proper_loc and proper_size :  
            # print('y_top:' + str(y_top) + ', y_bottom:' + str(y_bottom) + ', volum: '+str(volum) )
            # print('width:'+str(width)+ ', height:' + str(height) + ', length:' + str(length))
            bbox_select_list.append(aabb)
            pcd_filter.append(obj_cloud)
            label_filter += [label_ind] * len(index)
            label_ind += 1

    if visual:
        # o3d.visualization.draw_geometries(bbox_vis_list,
        #                             zoom=0.7,
        #                             front=[0.5439, -0.2333, -0.8060],
        #                             lookat=[2.4615, 2.1331, 1.338],
        #                             up=[-0.1781, -0.9708, 0.1608])

        o3d.visualization.draw_geometries(pcd_filter + bbox_select_list,
                                    zoom=0.7,
                                    front=[0.5439, -0.2333, -0.8060],
                                    lookat=[2.4615, 2.1331, 1.338],
                                    up=[-0.1781, -0.9708, 0.1608])

    return label_filter, pcd_filter 


def dbscan_cluster(pcd, visual=False): ## 聚类
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.7, min_points=7, print_progress=False))

    max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")

    if visual:
        colors = plt.get_cmap("hsv")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcd],
        #                                 zoom=0.8,
        #                                 front=[-0.4999, -0.1659, -0.8499],
        #                                 lookat=[2.1813, 2.0619, 2.0999],
        #                                 up=[0.1204, -0.9852, 0.1215])

    label_filter, pcd_filter = remove_non_object(labels, pcd, visual)

    return label_filter, pcd_filter


def test_ransac():
    range_xy = [-350,2300, 0, 1080]   ## x_min, x_max, y_min, y_max

    select_num = [9323, 8371, 10471,27653,29020]
    img_dir = '../dataset/hs5_0831/image_undistort'
    lidar_dir = '../dataset/hs5_0831/pcl_pcd'
    time_align_txt = '../dataset/hs5_0831/lidar_camera_align.txt'
    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()
    
    for ii, path in enumerate(align_paths):
        if ii in select_num:               #  筛选一帧目标多的点云
            pcd_path = os.path.join(lidar_dir, path.split()[0])
            raw_points = get_pcd(pcd_path)
            # vis_points(raw_points)

            # img_path = os.path.join(img_dir, path.split()[1])
            # img_name = path.split()[1].split('.')[0]
            # img = cv2.imread(img_path)
            # img = cv2.resize(img,(736,416))


            # ## 先滤地面，后筛点
            # point_cloud = remove_ground(raw_points, method = 'RANSAC', visual=True)
            # points_no_ground = np.asarray(point_cloud.points)
            # points_no_ground = np.hstack((points_no_ground, np.ones((points_no_ground.shape[0],1))))
            # points = point_in_img(points_no_ground, range_xy)
            # cloud = np2pcd(points)

            ## 先筛点，后滤地面
            points = point_in_img(raw_points, range_xy)
            # vis_points(points)
            cloud = remove_ground(points, method = 'RANSAC', visual=True)

            label_filter, pcd_filter = dbscan_cluster(cloud, visual=True)

            # label_, pcd_ = postprocess_cluster(pcd_filter)


def test_cylinder():
    range_xy = [-350,2300, 0, 1080]   ## x_min, x_max, y_min, y_max

    select_num = [9323, 8371, 10471, 27653, 29020]
    img_dir = '../dataset/hs5_0831/image_undistort'
    lidar_dir = '../dataset/hs5_0831/pcl_pcd'
    time_align_txt = '../dataset/hs5_0831/lidar_camera_align.txt'
    pred_label_path = '../dataset/hs5_0831/results_de_nusc_test'
    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()
    
    for ii, path in enumerate(align_paths):
        if ii in select_num:               #  筛选一帧目标多的点云
            pcd_path = os.path.join(lidar_dir, path.split()[0])
            raw_points = get_pcd(pcd_path, filter_method = 'cylinder')  #  坐标系转成nusc

            # 读取label信息
            pcd_name = pcd_path.split('/')[-1].split('.pcd')[0]
            label_ = os.path.join(pred_label_path, pcd_name+'.label')
            label_pred = np.fromfile(label_, dtype=np.uint32)
            label_pred = map_label(label_pred, label_mapping_nusc_2)

            filter_points = remove_ground_cyliner(raw_points, label_pred)  #  记得只要x>0的点
            vis_points(filter_points)

            ## 筛点
            points = point_in_img(filter_points, range_xy)
            cloud = np2pcd(points)
            label_filter, pcd_filter = dbscan_cluster(cloud, visual=True)


def run_cluster():
    filter_method = 'RANSAC'
    range_xy = [-350, 2300, 0, 1080]  ## x_min, x_max, y_min, y_max
    img_dir = '../dataset/hs5_0831/image_undistort'
    lidar_dir = '../dataset/hs5_0831/pcl_pcd'
    time_align_txt = '../dataset/hs5_0831/weitang_for_asso_lidar_cam_align.txt'
    pred_label_path = '../dataset/hs5_0831/results_weitang_for_asso'

    output_dir = 'output/clustering/vis_cluster_weitang_0831_cylinder_bbox'
    visualize = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()
    
    for ii, path in enumerate(align_paths):
        if ii > -1:
            pcd_path = os.path.join(lidar_dir, path.split()[0])
            
            if filter_method =='RANSAC':
                raw_points = get_pcd(pcd_path)
                points = point_in_img(raw_points, range_xy)
                cloud = remove_ground(points, method = 'RANSAC', visual=visualize)
                label_cluster, pcd_cluster = dbscan_cluster(cloud, visual=visualize)

            elif filter_method =='Cylinder':
                raw_points = get_pcd(pcd_path, filter_method = 'cylinder')  
                pcd_name = pcd_path.split('/')[-1].split('.pcd')[0]
                label_ = os.path.join(pred_label_path, pcd_name+'.label')
                label_pred = np.fromfile(label_, dtype=np.uint32)
                label_pred = map_label(label_pred, label_mapping_nusc_2)
                filter_points = remove_ground_cyliner(raw_points, label_pred) 
                # vis_points(filter_points)
                points = point_in_img(filter_points, range_xy)
                cloud = np2pcd(points)
                label_cluster, pcd_cluster = dbscan_cluster(cloud, visual=visualize)

            img_path = path.split()[1].strip()
            paint_cluster_on_img(img_path, img_dir, output_dir, pcd_cluster)

            print(str(ii) + '/' + str(len(align_paths)) +' is done.')


def main():
    test_ransac()
    # test_cylinder()
    # run_cluster()


if __name__ == '__main__':
    main()            