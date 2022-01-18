import os
import time

import numpy as np
import cv2
import vtk
import argparse

from scale_recovery import project_to_3d
from visual.vtk_wrapper.vtk_point_cloud_glyph import VtkPointCloudGlyph
from visual.vtk_wrapper import vtk_utils

def setup_vtk_renderer():
    # Setup renderer
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    return vtk_renderer

def setup_vtk_render_window(window_name, window_size, vtk_renderer):
    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(window_name)
    vtk_render_window.SetSize(*window_size)
    vtk_render_window.AddRenderer(vtk_renderer)
    return vtk_render_window

def get_velo_points(raw_data, frame_idx):
    velo_points = raw_data.get_velo(frame_idx)

    # Filter points to certain area
    points = velo_points[:, 0:3]
    area_extents = np.asarray([[0, 100], [-50, 50], [-5, 1]], dtype=np.float32)
    area_filter = \
        (points[:, 0] > area_extents[0, 0]) & \
        (points[:, 0] < area_extents[0, 1]) & \
        (points[:, 1] > area_extents[1, 0]) & \
        (points[:, 1] < area_extents[1, 1]) & \
        (points[:, 2] > area_extents[2, 0]) & \
        (points[:, 2] < area_extents[2, 1])
    points = points[area_filter]

    return points

def visual_depth_real_scale(depth_dir, img_dir, out_dir, width=736, height=416):

    
    scale_x, scale_y = 736/1920, 416/1080
    cx,cy,fx,fy = 851.8287*scale_x, 424.0041*scale_y, 1998.7356*scale_x, 1991.8909*scale_y

    vtk_window_size = (1280, 720)
    max_fps = 10000.0
    file_list = sorted(os.listdir(depth_dir))
    frame_range = (0, len(file_list))
    ##########################################
    min_loop_time = 1.0 / max_fps
    vtk_renderer = setup_vtk_renderer()
    vtk_render_window = setup_vtk_render_window('Overlaid Point Cloud', vtk_window_size, vtk_renderer)
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.SetViewUp(0, 0, 1)    # 0, -1, 0
    current_cam.SetPosition(0, -10, -20)  #  0.0, 0, 0
    current_cam.SetFocalPoint(0.0, 0.0, 20)
    current_cam.SetViewAngle(50)
    current_cam.Zoom(0.9)

    # # Create VtkAxes
    vtk_axes = vtk.vtkAxesActor()
    vtk_axes.SetTotalLength(1, 1, 1)

    # Setup interactor
    vtk_interactor = vtk.vtkRenderWindowInteractor()
    vtk_interactor.SetRenderWindow(vtk_render_window)
    vtk_interactor.SetInteractorStyle(vtk_utils.ToggleActorsInteractorStyle(None, vtk_renderer, current_cam, vtk_axes))
    vtk_interactor.Initialize()

    # Point cloud
    vtk_pc = VtkPointCloudGlyph()

    # # Add actors
    vtk_renderer.AddActor(vtk_pc.vtk_actor)

   ###########################################
    for frame_idx in range(*frame_range):
        loop_start_time = time.time()
        print('{} / {}'.format(frame_idx, len(file_list) - 1))
        
        # Load next frame data
        load_start_time = time.time()

        frame = file_list[frame_idx]
        depth_map_path = os.path.join(depth_dir, frame)
        depth_map = np.load(depth_map_path)  # 736*416

        img_name = frame.split('.')[0]
        img_path = img_name +'.jpg'
        raw_image_path = os.path.join(img_dir, img_path)
        bgr_image = cv2.imread(raw_image_path)
        bgr_image = cv2.resize(bgr_image, (width,height))
        # print(raw_image_path)
        cam0_curr_pc = project_to_3d(depth_map, cx, cy, fx, fy, width, height)
        point_colours = bgr_image.reshape(-1, 3)

        print('load\t\t', time.time() - load_start_time)

        # VtkPointCloud
        vtk_pc_start_time = time.time()
        vtk_pc.set_points(cam0_curr_pc, point_colours)
        print('vtk_pc\t\t', time.time() - vtk_pc_start_time)

        # Reset the clipping range to show all points
        vtk_renderer.ResetCameraClippingRange()

        out_depth_map_path = os.path.join(out_dir, img_name + '.png')
        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput(vtk_render_window)
        grabber.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetInputData(grabber.GetOutput())
        writer.SetFileName(out_depth_map_path)
        writer.Write()

        # Render
        render_start_time = time.time()
        vtk_render_window.Render()
        print('render\t\t', time.time() - render_start_time)

        # Pause to keep frame rate under max
        loop_run_time = time.time() - loop_start_time
        print('loop\t\t', loop_run_time)
        if loop_run_time < min_loop_time:
            time.sleep(min_loop_time - loop_run_time)

        print('---')

    print('Done')

    # Keep window open
    vtk_interactor.Start()


def main():
    depth_dir = '../dataset/hs5_0831/depth_map/hs5_35k_3_east_park' 
    img_dir = '../dataset/hs5_0831/image_undistort'
    out_dir = './output/hs5_35k_3_east_park'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    width = 736
    height = 416
    visual_depth_real_scale(depth_dir, img_dir, out_dir, width, height)

if __name__ == '__main__':
    main()
