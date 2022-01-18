# -*- coding: utf-8 -*-
# Construct a concave or convex hull polygon for a plane model
# http://pointclouds.org/documentation/tutorials/hull_2d.php#hull-2d

import numpy as np
# import pcl
# import random
# from calib_lidar_cam import filter_passthrough , visual_cloud
from visual.vis_o3d import vis_points, np2pcd

# def create_hull():
#     lidar_points = np.load('calib_data/lidar_points_trun20_transformed.npy').astype(np.float32)
#     lidar_points = filter_passthrough(lidar_points)
#     cloud_l = pcl.PointCloud()
#     cloud_l.from_array(lidar_points)
#     visual_cloud(cloud_l)
#     # // Build a filter to remove spurious NaNs
#     print('PointCloud after filtering has: ' +str(cloud_l.size) + ' data points.')

#     #   // Create a Concave Hull representation of the projected inliers
#     chull = cloud_l.make_ConcaveHull()
#     chull.set_Alpha(0.1)
#     cloud_hull = chull.reconstruct()
#     print('Concave hull has: ' + str(cloud_hull.size) + ' data points.')
#     visual_cloud(cloud_hull)

import open3d as o3d


def filter_passthrough(points):
    filter_points = list()
    for point in points:
        if abs(point[0]) < 4 and point[2] > 5 and point[1]> -0.5:
            filter_points.append(point)
    return np.array(filter_points)


def construct_convex_hull():
    lidar_points = np.load('calib_data/lidar_points_trun20_transformed.npy').astype(np.float32)
    lidar_points = filter_passthrough(lidar_points)
    lidar_points = np.vstack((lidar_points, np.array([[0,-1,10]])))
    pcd = np2pcd(lidar_points)
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.664,
                                    front=[-0.4761, -0.4698, -0.7434],
                                    lookat=[1.8900, 3.2596, 0.9284],
                                    up=[0.2304, -0.8825, 0.4101])

    # # 估计法线
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # radii = [0.005, 0.01, 0.02, 0.04]
    # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd, rec_mesh])

    # print('run Poisson surface reconstruction')
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh],
    #                                 zoom=0.664,
    #                                 front=[-0.4761, -0.4698, -0.7434],
    #                                 lookat=[1.8900, 3.2596, 0.9284],
    #                                 up=[0.2304, -0.8825, 0.4101])

def main():
    # create_hull()
    construct_convex_hull()

if __name__ == "__main__":
    main()