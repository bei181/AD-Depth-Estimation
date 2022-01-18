# -*- coding: utf-8 -*-
# How to use iterative closest point
# http://pointclouds.org/documentation/tutorials/iterative_closest_point.php#iterative-closest-point

import pcl
import random
import numpy as np

import pcl.pcl_visualization
from ICP.util_icp import filter_passthrough, turbulent

def visual_cloud_pair(cloud1, cloud2):
    visualcolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud2, 0, 255, 0)
    visualcolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud1, 255,0,0)
    vs = pcl.pcl_visualization.PCLVisualizering
    vss1 = pcl.pcl_visualization.PCLVisualizering() #初始化一个对象，这里是很重要的一步
    
    vs.AddPointCloud_ColorHandler(vss1, cloud1, visualcolor2, id=b'cloud1', viewport=0)
    vs.AddPointCloud_ColorHandler(vss1, cloud2, visualcolor1,id=b'cloud2',viewport=0)
    vss1.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 2, b'cloud1')#设置点的大小
    vss1.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud2')#设置点的大小

    while not vs.WasStopped(vss1):
       vs.Spin(vss1)

def visual_cloud(cloud):
    visual = pcl.pcl_visualization.CloudViewing()
    visual.ShowMonochromeCloud(cloud)
    flag = 'True'
    while flag:
        flag = not(visual.WasStopped())

def run_icp(points_p, points_l):
    cloud_p = pcl.PointCloud()
    cloud_l = pcl.PointCloud()
    cloud_p.from_array(points_p)
    cloud_l.from_array(points_l)
    print('Transformed ' + str(cloud_p.size) + ' data points:')
    # visual_cloud_pair(cloud_p, cloud_l)

    # 对pseudo_lidar下采样
    sor = cloud_p.make_voxel_grid_filter()
    sor.set_leaf_size(0.2, 0.1, 0.2)
    cloud_filtered_p = sor.filter()
    visual_cloud_pair(cloud_filtered_p,cloud_l)

    # start ICP
    icp = cloud_l.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(cloud_l, cloud_filtered_p)
    print('has converged:' + str(converged) + ' score: ' + str(fitness))
    print(str(transf))
    return transf


def example():
    cloud_in = pcl.PointCloud()
    cloud_out = pcl.PointCloud()

    # Fill in the CloudIn data
    # cloud_in->width    = 5;
    # cloud_in->height   = 1;
    # cloud_in->is_dense = false;
    # cloud_in->points.resize (cloud_in->width * cloud_in->height);
    # for (size_t i = 0; i < cloud_in->points.size (); ++i)
    # {
    #   cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
    #   cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
    #   cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
    # }
    points_in = np.zeros((5, 3), dtype=np.float32)
    RAND_MAX = 1024.0
    for i in range(0, 5):
        points_in[i][0] = 1024 * random.random() / RAND_MAX
        points_in[i][1] = 1024 * random.random() / RAND_MAX
        points_in[i][2] = 1024 * random.random() / RAND_MAX

    cloud_in.from_array(points_in)

    # std::cout << "Saved " << cloud_in->points.size () << " data points to input:" << std::endl;
    # for (size_t i = 0; i < cloud_in->points.size (); ++i) std::cout << "    " <<
    #   cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<
    #   cloud_in->points[i].z << std::endl;
    # *cloud_out = *cloud_in;
    print('Saved ' + str(cloud_in.size) + ' data points to input:')
    points_out = np.zeros((5, 3), dtype=np.float32)

    # std::cout << "size:" << cloud_out->points.size() << std::endl;
    # for (size_t i = 0; i < cloud_in->points.size (); ++i)
    # cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;

    # print('size:' + str(cloud_out.size))
    # for i in range(0, cloud_in.size):
    print('size:' + str(points_out.size))
    for i in range(0, cloud_in.size):
        points_out[i][0] = points_in[i][0] + 0.7
        points_out[i][1] = points_in[i][1]
        points_out[i][2] = points_in[i][2]

    cloud_out.from_array(points_out)

    # std::cout << "Transformed " << cloud_in->points.size () << " data points:" << std::endl;
    print('Transformed ' + str(cloud_in.size) + ' data points:')

    # for (size_t i = 0; i < cloud_out->points.size (); ++i)
    #   std::cout << "    " << cloud_out->points[i].x << " " << cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
    for i in range(0, cloud_out.size):
        print('     ' + str(cloud_out[i][0]) + ' ' + str(cloud_out[i]
                                                         [1]) + ' ' + str(cloud_out[i][2]) + ' data points:')

    # pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    # icp.setInputCloud(cloud_in);
    # icp.setInputTarget(cloud_out);
    # pcl::PointCloud<pcl::PointXYZ> Final;
    # icp.align(Final);
    icp = cloud_in.make_IterativeClosestPoint()
    # Final = icp.align()
    converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)

    # std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    # std::cout << icp.getFinalTransformation() << std::endl;
    # print('has converged:' + str(icp.hasConverged()) + ' score: ' + str(icp.getFitnessScore()) )
    # print(str(icp.getFinalTransformation()))
    print('has converged:' + str(converged) + ' score: ' + str(fitness))
    print(str(transf))


def verify_trans(lidar2camera_matrix, pseudo_points, lidar_points):
    lidar_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0],1))))
    camera_coor_lidar_points = np.matmul(lidar2camera_matrix, lidar_points.transpose())[:3,:]
    cloud_p = pcl.PointCloud()
    cloud_l = pcl.PointCloud()
    cloud_p.from_array(pseudo_points)
    cloud_l.from_array(camera_coor_lidar_points.transpose().astype(np.float32))
    
    # 对pseudo_lidar下采样
    sor = cloud_p.make_voxel_grid_filter()
    sor.set_leaf_size(0.2, 0.1, 0.2)
    cloud_filtered_p = sor.filter()
    visual_cloud_pair(cloud_filtered_p,cloud_l)



def main():
    pseudo_points = np.load('calib_data/pseudo_lidar_trun20.npy').astype(np.float32)
    lidar_points = np.load('calib_data/lidar_points_trun20_transformed.npy').astype(np.float32)
    pseudo_points = filter_passthrough(pseudo_points)
    lidar_points = filter_passthrough(lidar_points)

    # 加小扰动
    rotation = [0.1,0.05,0.05]
    transation = [0.1,0.05,0.05]
    lidar_points = turbulent(lidar_points,rotation,transation)

    trans_matrix = run_icp(pseudo_points, lidar_points)
    verify_trans(trans_matrix, pseudo_points, lidar_points)



if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()