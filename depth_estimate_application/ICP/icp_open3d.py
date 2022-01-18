import open3d as o3d
import numpy as np
import copy
import math
from sklearn.neighbors import KDTree

from ICP.util_icp import filter_passthrough, turbulent
from clustering.cluster import np2pcd


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def get_init_trans(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # print(result_ransac)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_fast = execute_fast_global_registration(source_down, target_down,
                                               source_fpfh, target_fpfh,
                                               voxel_size)
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)

    result_icp = refine_registration(source, target, result_fast, voxel_size)
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

    return result_icp.transformation


def filter_pseudo_by_lidar(pseudo_points, lidar_points, r_thres=1):
    both_points = np.concatenate((lidar_points,pseudo_points))
    tree = KDTree(both_points, leaf_size=2)   
    neighbor_ind = tree.query_radius(both_points, r=r_thres)

    N_L = lidar_points.shape[0]
    N_PL = pseudo_points.shape[0]
    ind_PL = neighbor_ind[N_L:]

    fliter_pseudo_points = list()
    for ii,ind in enumerate(ind_PL):
        if min(ind) < N_L:  # 半径范围内有lidar点
            fliter_pseudo_points.append(pseudo_points[ii])

    return np.array(fliter_pseudo_points)


def run_icp(pseudo_points, lidar_points, visual = True):
    threshold = 10
    voxel_size = 0.05
    r_filter = 1

    cloud_p = np2pcd(pseudo_points)
    cloud_l = np2pcd(lidar_points)
    cloud_p = cloud_p.voxel_down_sample(0.15)

    # 去除伪激光的拖影点
    filtered_pseudo_points = filter_pseudo_by_lidar(np.array(cloud_p.points), lidar_points,r_thres = r_filter)
    cloud_p = np2pcd(filtered_pseudo_points)

    # 配准前评价
    trans_init = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    # trans_init = get_init_trans(cloud_l, cloud_p, voxel_size)
    draw_registration_result(cloud_l, cloud_p,trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(cloud_l, cloud_p, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(cloud_l, cloud_p, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling =True),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    # 找尺度因子，矫正伪激光点云的尺度
    cR_matrix = reg_p2p.transformation[0:3,0:3]
    c2I_matrix = np.matmul(cR_matrix.transpose(),cR_matrix)
    c_scale = math.sqrt(c2I_matrix[0,0])
    scale_p = 1/c_scale
    cloud_p.scale(scale_p, np.array([0,0,0]))

    # 去除尺度因子后的变换矩阵
    transformation = reg_p2p.transformation.copy()
    R_ = cR_matrix / c_scale
    transformation[0:3,0:3] = R_
    print("Transformation in real scale is:")
    print(transformation)
    source_transformed = draw_registration_result(cloud_l, cloud_p, transformation, visual = visual, return_res = True)

    return source_transformed, scale_p, transformation


def draw_registration_result(source, target, transformation = None, visual= True, return_res=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    if visual:
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])
    if return_res:
        return source_temp


def point_registration(pseudo_points,lidar_points, visual = True):
    pseudo_points = filter_passthrough(pseudo_points.astype(np.float32))
    lidar_points = filter_passthrough(lidar_points.astype(np.float32))

    lidar_cloud_icp, scale_p, transformation = run_icp(pseudo_points, lidar_points,visual = visual)
    lidar_points_icp = np.array(lidar_cloud_icp.points)

    return lidar_points_icp, scale_p, transformation


def main():
    pseudo_points = np.load('calib_data/points_npy/pseudo_lidar_trun20.npy').astype(np.float32)
    lidar_points = np.load('calib_data/points_npy/lidar_points_trun20_transformed.npy').astype(np.float32)
    pseudo_points = filter_passthrough(pseudo_points)
    lidar_points = filter_passthrough(lidar_points)

    # # 加小扰动
    # rotation = [0.1,0.05,0.05]
    # transation = [0.1,0.05,0.05]
    # lidar_points = turbulent(lidar_points,rotation,transation)

    lidar_cloud_icp,_,_ = run_icp(pseudo_points, lidar_points)



if __name__ == '__main__':
    main()