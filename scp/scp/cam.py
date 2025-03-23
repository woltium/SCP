from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix,get_real_depth_map
def capture_current_point_cloud(physics, height, width, camera_name):
    """
    Captures point cloud from the current camera view.

    Args:
        physics (MjSim): MuJoCo physics simulator instance
        height (int): Height of camera image in pixels
        width (int): Width of camera image in pixels  
        camera_name (str): Name of camera to use

    Returns:
        points (np.array): Point cloud array of shape (H, W, 3) containing 3D points in world coordinates
    """
    # Get camera matrices
    camera_transform = get_camera_transform_matrix(
        sim=physics,
        camera_name=camera_name,
        camera_height=height,
        camera_width=width
    )
    camera_to_world = np.linalg.inv(camera_transform)

    # Get depth map
    map = physics.render(
        camera_name=camera_name,
        height=height,
        width=width,
        depth=True
    )
    depth_map = map[1]
    rgb = map[0]
    # Convert normalized depth to real depth
    depth_map = get_real_depth_map(physics, depth_map)
    
    # Create pixel coordinate grid
    pixel_x = np.arange(0, width)
    pixel_y = np.arange(0, height)
    pixel_coords = np.meshgrid(pixel_x, pixel_y)
    
    # Stack coordinates and reshape
    pixels = np.stack((pixel_coords[0], pixel_coords[1]), axis=-1)
    pixels = pixels.reshape(-1, 2)
    
    # Add homogeneous coordinate
    pixels_homog = np.column_stack((pixels, np.ones(len(pixels))))
    
    # Get 3D points in camera frame
    points_cam = pixels_homog @ camera_transform[:3, :3].T
    points_cam = points_cam * depth_map.reshape(-1, 1)
    
    # Transform to world frame
    points_homog = np.column_stack((points_cam, np.ones(len(points_cam))))
    points = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Reshape back to image dimensions
    points = points.reshape(height, width, 3)


    
    return points
