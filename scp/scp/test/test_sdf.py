import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
from environment import ReKepMujocoEnv
from scp.utils.rekep_utils import get_config
from camera_utils import capture_current_point_cloud
# Load config
global_config = get_config(config_path="./configs/config.yaml")


# Create environment with camera segmentation enabled
env = suite.make(
    env_name="Lift",
    robots="Panda", 
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_segmentations=["instance", "class", "element"],
    camera_heights=512,
    camera_widths=512,
    camera_depths=True
)


obs = env.reset()
# Initialize the environment wrapper
env_wrapper = ReKepMujocoEnv(config=global_config['env'], env=env, verbose=True)

# Get SDF voxels
resolution = 0.01  # 1cm resolution
sdf_voxels = env_wrapper.get_sdf_voxels(resolution=resolution)

# Visualize the SDF
env_wrapper.visualize_sdf(
    sdf_voxels, 
    env_wrapper.bounds_min, 
    env_wrapper.bounds_max, 
    resolution
)