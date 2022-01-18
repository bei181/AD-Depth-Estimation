import numpy as np
import math


def filter_passthrough(points):
    filter_points = list()
    for point in points:
        if point[2] > 5 and point[1]> -0.59: # abs(point[0]) < 4 and point[1]> -1 and point[2] > 5 :
            filter_points.append(point)
    return np.array(filter_points)


def unstacked_matrix_from_angles(rx, ry, rz):
  """Create an unstacked rotation matrix from rotation angles.
  Args:
    rx: rotation angles abound x, of any shape.
    ry: rotation angles abound y (of the same shape as x).
    rz: rotation angles abound z (of the same shape as x).
  Returns:
    A 3 * 3 numpy array , representing the respective rotation matrix.
  """
  angles = [-rx, -ry, -rz]
  sx, sy, sz = [math.sin(a) for a in angles]
  cx, cy, cz = [math.cos(a) for a in angles]
  m00 = cy * cz
  m10 = (sx * sy * cz) - (cx * sz)
  m20 = (cx * sy * cz) + (sx * sz)
  m01 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m21 = (cx * sy * sz) - (sx * cz)
  m02 = -sy
  m12 = sx * cy
  m22 = cx * cy
  return np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])


def turbulent(lidar_points,rotation,transation):
    rotation_matrix = unstacked_matrix_from_angles(rotation[0], rotation[1], rotation[2])
    transation_matrix = np.array([[transation[0]],[transation[1]],[transation[2]]])
    trans_matrix = np.hstack((rotation_matrix,transation_matrix))
    trans_matrix = np.vstack((trans_matrix,np.array([[0,0,0,1]])))

    lidar_points = np.hstack((lidar_points, np.ones((lidar_points.shape[0],1))))
    turbu_lidar_points = np.matmul(trans_matrix, lidar_points.transpose())[:3,:]
    turbu_lidar_points = turbu_lidar_points.transpose().astype(np.float32)
    return turbu_lidar_points

