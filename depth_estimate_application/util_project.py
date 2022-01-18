import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

############ Tuned Parameters by hand ######################################################################
raw_width, raw_height = 1920, 1080
img_width_scaled, img_height_scaled = 736,416
scale_x, scale_y = img_width_scaled/raw_width, img_height_scaled/raw_height

Lidar2CamMatrix = np.array([[0.08083690,      -0.99662126,    -0.01167763,    -0.19623674],
                                [0.04087759,     0.01505343,     -0.99906721,    -0.26569012],
                                [0.99591426,     0.08030508,     0.04197840,     -0.59186737],
                                [0,0,0,1]])

CameraIntrinsics_scaled = np.array([[1998.7356*scale_x,    0,   851.8287*scale_x,],
                                [0,    1991.8909*scale_y,   424.0041*scale_y]])

CameraIntrinsics = np.array([[1998.7356,    0,   851.8287,],
                                [0,    1991.8909,   424.0041]])

############################################################################################################

Lidar2CamMatrix_new = np.array([[0.04623771756705647, -0.998929491730468, 0.001394283076640634, -0.01006832786180179],
                                [0.005611990133126521, -0.001135990693398981, -0.9999836074115864, -0.1452161342924859],
                                [0.998914700583055, 0.04624478431405227, 0.005553456834440085, -0.6479335759534361],
                                [0, 0, 0, 1]])
                                    
CameraIntrinsics_new = np.array([[964.5141731310823 , 0, 953.4057393830921], 
                                [0, 840.5867921056233 , 540.4464063044832]], dtype=np.float32)
############################################################################################################

def project_to_3d(depth_recover, cx, cy, fx, fy,width, height, truncate=False):
    # points = np.ones((1080, 1920, 4))
    points = np.ones((height, width, 3))
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    X = (X-cx)/fx
    Y = (Y-cy)/fy

    # depth_recover = np.squeeze(depth_recover,axis=2)
    depth_recover_mul = np.stack((depth_recover,depth_recover,depth_recover),axis=2)
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
        points = np.array(new_points).reshape(-1,3)


    return points

def get_pcd(file_path, filter_method=None):
    pts = []
    
    f = open(file_path, 'r')
    data = f.readlines()
    f.close()
    line = data[9]
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])

    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        x, y, z, intensity = [eval(i) for i in xyzi]

        if not filter_method:
            if x <= 0.01:
                continue
            pts.append([float(x), float(y), float(z), 1])  # 转成齐次坐标
        elif filter_method == 'cylinder':
            pts.append([float(x), float(y), float(z), 1])  # 保留所有的点

    # except:
    #     f = open(file_path, 'rb')
    #     data = f.readlines()
    #     f.close()
    #     line = data[9].decode()
    #     line = line.strip('\n')
    #     i = line.split(' ')
    #     pts_num = eval(i[-1])

    #     for line in data[11:]:
    #         line = line.decode()
    #         line = line.strip('\n')
    #         xyzi = line.split(' ')
    #         x, y, z, intensity = [eval(i) for i in xyzi]
    #         if x <= 0.01:
    #             continue
    #         pts.append([float(x), float(y), float(z), 1])  # 转成齐次坐标

    points = np.array(pts)
    return points  # shape:(N,4)


def lidar_to_pixel(points, CameraIntrin=CameraIntrinsics_scaled, Lidar2Cam=Lidar2CamMatrix, width=736, height=416, filter_img_point=True, z_range=55): # shape:(N,4)
    camera_coor_points = np.matmul(Lidar2Cam, points.transpose())[:3,:] # shape:(3,N)
    camera_coor_points_norm = camera_coor_points / camera_coor_points[2,:]  # Normalization, shape:(3,N)
    pixel_points = np.matmul(CameraIntrin, camera_coor_points_norm) # shape:(2,N)
    pixel_points = pixel_points.transpose() # shape:(N,2)

    # 超出图像范围的点丢掉
    lidar2pixel_points, lidar2pixel_dpoints = list(),list()
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i,:]
        z = camera_coor_points[2,i]  # 目标点到相机的深度（z方向的距离）

        if filter_img_point:
            if x > 0  and x < width and  y > 0 and y < height and z < z_range:
                lidar2pixel_points.append((int(x),int(y)))
                lidar2pixel_dpoints.append((int(x),int(y),z))
        else:
            lidar2pixel_points.append((int(x),int(y)))
            lidar2pixel_dpoints.append((int(x),int(y),z))

    return lidar2pixel_points, lidar2pixel_dpoints


def camera_to_pixel(camera_coor_points, CameraInt, width, height): # shape:(3,N)
    camera_coor_points_norm = camera_coor_points / camera_coor_points[2,:]  # Normalization, shape:(3,N)
    pixel_points = np.matmul(CameraInt, camera_coor_points_norm) # shape:(2,N)
    pixel_points = pixel_points.transpose() # shape:(N,2)
    # 超出图像范围的点丢掉
    cam2pixel_points = list()
    for i in range(pixel_points.shape[0]):
        x, y = pixel_points[i,:]
        if x > 0  and x < width and  y > 0 and y < height:
            cam2pixel_points.append((int(x),int(y)))
    return cam2pixel_points


def draw_points(points, img):
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 1            # 可以为 0 、4、8
    for point in points:
        cv2.circle(img, point, point_size, point_color, thickness)
    return img

def draw_points_with_label(points, img, color, draw_box =True):
    point_size = 1
    thickness = 4             # 可以为 0 、4、8
    for point in points:
        cv2.circle(img, point, point_size, color, thickness)
    
    if draw_box and len(points) > 0:
        p_array = np.array(points)
        c1 = (np.min(p_array[:,0]), np.min(p_array[:,1]))
        c2 = (np.max(p_array[:,0]), np.max(p_array[:,1]))
        cv2.rectangle(img, c1, c2, color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def project_3d_2d(time_align_txt, img_dir, pcd_dir, out_dir, img_range, CameraInt, Lidar2Cam, width, height, z_range):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(time_align_txt,'r') as f_read:
        align_paths = f_read.readlines()
 
    for ii, path in enumerate(align_paths):
        if ii >= int(img_range[0]):
            print(ii)
            pcd_path = os.path.join(pcd_dir, path.split()[0])
            img_path = os.path.join(img_dir, path.split()[1])
            points = get_pcd(pcd_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (width, height))
            lidar_points, lidar_dpoints = lidar_to_pixel(points, CameraInt, Lidar2Cam, width=width, height=height, z_range=z_range)
            img = draw_points(lidar_points, img)
            cv2.imwrite(os.path.join(out_dir, path.split()[1].split('.')[0] + '_fusion_part.png'), img)
        if ii > int(img_range[1]):
            break


def project_pc_2d(img, points, save_path, CameraInt, width, height):
    lidar_points = camera_to_pixel(points, CameraInt, width=width, height=height)
    img = draw_points(lidar_points,img)
    cv2.imwrite(save_path, img)


## 把ground 类视为：待去除的地面点
label_mapping_nusc_2={
    ## ground
    26:0,   # "sidewalk"
    24:0,   # 'flat.driveable_surface'
    25:0,   # 'flat.other',
    ## vehicle
    17:1,   # "car"
    14:1,   # "bicycle"
    21:1,   # "motorcycle"
    23:1,   # "truck"
    18:1,   # "construction_vehicle"
    16:1,   # "bus"
    22:1,   # 'trailer'
    ## nature
    30:1,   # "vegetation"
    27:1,   # "terrain"
    ## human
    2:1,    # "person"
    ## object
    9:1,    # "barrier"
    12:1,   # "traffic cone"
    28:1,   # "static_manmade"
    ## others
    0:1,    # "unlabeled", and others ignored
}

def map_label(pred, label_mapping):
    temp = pred.copy()
    for v, k in label_mapping.items():
        pred[temp == v] = k
    return pred


def paint_cluster_on_img(img_name, img_dir, output_dir, pcd_cluster):
    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)
    max_label = len(pcd_cluster)+1

    for ii, cluster_p in enumerate(pcd_cluster): 
        color = plt.get_cmap("hsv")(ii / (max_label))
        color_255 = np.array(color[0:3]) *255
        points = np.asarray(cluster_p.points).transpose()
        pixel_points = camera_to_pixel(points, CameraIntrinsics_scaled,raw_width,raw_height) # shape:(3,N)
        image = draw_points_with_label(pixel_points, image, color_255)

    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path,image)


def main():
    time_align_txt = '../dataset/hs5_0831/online_calib.txt' 
    img_dir = '../dataset/hs5_0831/image_undistort'
    pcd_dir = '../dataset/hs5_0831/pcl_pcd'
    out_dir = 'output/calib_data/project_lidar_0831_calibb'
    img_range = np.array([0,2])

    # 将lidar投影到image上
    project_3d_2d(time_align_txt, img_dir, pcd_dir, out_dir, img_range, CameraIntrinsics_scaled, 
                    Lidar2CamMatrix, img_width_scaled, img_height_scaled,z_range = 20) 

if __name__ == '__main__':
    main()
