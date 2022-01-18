from pyntcloud import PyntCloud
from pandas import DataFrame
import open3d as o3d
import numpy as np

def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

def vis_points(points):
    # lidar_points
    points = DataFrame(points[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    point_cloud_o3d_lidar = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    point_cloud_o3d_lidar.paint_uniform_color(np.array([[0],[0],[1]]))
    point_cloud_o3d_lidar.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([point_cloud_o3d_lidar], key_to_callback)


def vis_points_pair(source, target, visual= True):
    source = np2pcd(source)
    target = np2pcd(target)
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    if visual:
        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        o3d.visualization.draw_geometries_with_key_callbacks([source, target], key_to_callback)
        # o3d.visualization.draw_geometries([source, target],
        #                                 zoom=0.4459,
        #                                 front=[0.9288, -0.2951, -0.2242],
        #                                 lookat=[1.6784, 2.0612, 1.4451],
        #                                 up=[-0.3402, -0.9189, -0.1996])
 

def np2pcd(points):
    points = DataFrame(points[:, 0:3])  
    points.columns = ['x', 'y', 'z'] 
    point_cloud_pynt = PyntCloud(points) 
    point_cloud_o3d_lidar = point_cloud_pynt.to_instance("open3d", mesh=False) 
    point_cloud_o3d_lidar.paint_uniform_color(np.array([[0],[0],[1]]))
    return point_cloud_o3d_lidar


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
