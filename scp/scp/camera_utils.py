from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
#import robosuite.utils.camera_utils as CU
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_real_depth_map 

def show_depth_images(depth):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(depth[0])
    plt.colorbar()
    plt.title('Depth Image 0')
    
    plt.subplot(122)
    plt.imshow(depth[1])
    plt.colorbar()
    plt.title('Depth Image 1')
    
    plt.show()

def print_point_depth_stats(points):
 # Get points within specified range
 mask = (points[:,:,0] >= -0.1) & (points[:,:,0] <= 0.1) & \
        (points[:,:,1] >= -0.1) & (points[:,:,1] <= 0.1) & \
        (points[:,:,2] >= 0.8) & (points[:,:,2] <= 0.9)
    
 # Count points in range
 count = np.sum(mask)
    
 # Print result
 print(f"Points in range [[-0.1,0.1],[-0.1,0.1],[0.8,0.9]]: {count}")

def show_point_cloud(points):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reshape points to 2D array
    points_2d = points.reshape(-1, 3)
    
    # Subsample points by taking every 10th point
    points_2d = points_2d[::10]
    
    # Plot the points
    ax.scatter(points_2d[:, 0], points_2d[:, 1], points_2d[:, 2], s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



def capture_current_point_cloud(physics,height,width, camera_name):
    def depth_to_pointcloud(depth, intr, extr ):
        # 网格索引
        real_depth = get_real_depth_map(physics, depth)
        cc, rr = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), sparse=True)
        valid = (real_depth > 0)
        z = np.where(valid, real_depth, 0) 
        x = z * (cc - intr[0, 2]) / intr[0, 0]
        y = z * (rr - intr[1, 2]) / intr[1, 1]
        # 形成 (H, W, 3) 的点云
        xyz = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        #show_point_cloud(xyz)
        # 将点云坐标转换到世界坐标系
        xyz_h = np.concatenate([xyz, np.ones((*xyz.shape[:2], 1))], axis=-1)  # (H, W, 4)
        xyz_t = (extr @ xyz_h.reshape(-1, 4).T).T  # 通过外参矩阵转换 (N, 4)
        return xyz_t[:, :3].reshape(depth.shape[0], depth.shape[1], 3)  # (H, W, 3)

    # 获取相机 ID
    camera_id = physics.model.camera_name2id(camera_name)

    # 计算内参
    fov = physics.model.cam_fovy[camera_id]
    theta = np.deg2rad(fov)
    fx = width / 2 / np.tan(theta / 2)
    fy = height / 2 / np.tan(theta / 2)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 渲染深度图
    depth = physics.render(height=height, width=width, camera_name=camera_name, depth=True)
    depth = depth[1]
    # rotation_180_x = np.array([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1]
    #     ])

    # # 应用旋转修正
    # cam_pos = physics.data.cam_xpos[camera_id]
    # cam_rot = physics.data.cam_xmat[camera_id].reshape(3, 3)
    # cam_rot = rotation_180_x @ cam_rot 
    # extr = np.eye(4)
    # extr[:3, :3] = cam_rot.T
    # extr[:3, 3] = cam_pos
    extr = get_camera_extrinsic_matrix(physics,camera_name)
    #print(extr)
    #print("campos:",cam_pos)
    # 转换深度图为点云 (H, W, 3)
    points = depth_to_pointcloud(depth, intr, extr)
    #print('points:',points)
    #print_point_depth_stats(points)
    return points