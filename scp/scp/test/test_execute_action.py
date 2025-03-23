import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
from environment import ReKepMujocoEnv
from scp.utils.rekep_utils import get_config
from camera_utils import capture_current_point_cloud
from keypoint_proposal import KeypointProposer
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

# Load config
global_config = get_config(config_path="./configs/config.yaml")

controller_name = "OSC_POSE"
controller_config = load_controller_config(default_controller=controller_name)
controller_config['control_delta'] = False
# Create environment with camera segmentation enabled
env = suite.make(
    env_name="Lift",
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_segmentations=["instance", "class", "element"],
    camera_heights=512,
    camera_widths=512,
    camera_depths=True,
    controller_configs=controller_config
)

obs = env.reset()
# Initialize the environment wrapper
env_wrapper = ReKepMujocoEnv(config=global_config['env'], env=env, verbose=True)


# Get RGB image and create point cloud
rgb = obs['agentview_image']
depth = obs['agentview_depth']
points = capture_current_point_cloud(
    physics=env.sim,
    height=512,
    width=512,
    camera_name="agentview"
)
mask = obs['agentview_segmentation_element'][:, :, 0]
global_config = get_config(config_path="./configs/config.yaml")
keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
# Get keypoints using keypoint proposer
keypoints, projected_img = keypoint_proposer.get_keypoints(rgb, points, mask)

# Register keypoints
env_wrapper.register_keypoints(keypoints)

# Get current keypoint positions
current_keypoints = env_wrapper.get_keypoint_positions()
print("Current keypoint positions:", current_keypoints)
# Move to keypoint with z < 1
valid_keypoints = [kp for kp in current_keypoints if kp[2] < 1]

if len(valid_keypoints) > 0:
    target_pose = np.zeros(8)  # [x,y,z, qw,qx,qy,qz, gripper]
    target_pose[:3] = valid_keypoints[0]  # Position of first valid keypoint
    target_pose[3:7] = env_wrapper.get_ee_quat()  # Keep current orientation
    target_pose[7] = env_wrapper.get_gripper_open_action()  # Keep gripper open
print('target',target_pose)
print('axis',T.quat2axisangle(target_pose[3:7]))
env_wrapper._move_to_waypoint(target_pose)
target_pose[7]= env_wrapper.get_gripper_close_action()
env_wrapper._move_to_waypoint(target_pose)
for i in range(50):
    env_wrapper._step(target_pose)
print("done")
env.close()