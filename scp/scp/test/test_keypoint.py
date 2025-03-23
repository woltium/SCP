import robosuite as suite
import numpy as np
import matplotlib.pyplot as plt
from keypoint_proposal import KeypointProposer
from utils.rekep_utils import get_config
from camera_utils import capture_current_point_cloud
# Load config
import mimicgen
global_config = get_config(config_path="./configs/config.yaml")
keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
camera_name = 'agentview'
task = "Threading"
#task_prompt = global_config['task_descriptions'][task]['description']
# Create environment with camera segmentation enabled
env = suite.make(
    env_name=task,
    robots="Panda", 
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names = camera_name,
    camera_segmentations=["instance", "class", "element"],
    camera_heights=512,
    camera_widths=512,
    camera_depths=True
)

# Reset the environment
obs = env.reset()

# Get box position
# box_pos = env.sim.data.body_xpos[env.cube_body_id]
# print("Box position:", box_pos)



# Get RGB image and create point cloud
rgb = obs[camera_name+'_image']
#save the rgb image in jpg:
plt.imsave('rgb_nut.jpg', rgb)

depth = obs[camera_name+'_depth']# Get depth image
camera_fov = env.sim.model.cam_fovy[env.sim.model.camera_name2id(camera_name)]
#plt.imshow(rgb)
# Plot RGB and depth images side by side
#plt.show()
points = capture_current_point_cloud(
    physics=env.sim,
    height=512,
    width=512,
    camera_name=camera_name
)
mask = obs[camera_name+'_segmentation_element'][:, :, 0]

print('shape of points', points.shape)
print('shape of mask', mask.shape)
# Get keypoints using keypoint proposer
keypoints, projected_img = keypoint_proposer.get_keypoints(rgb, points, mask,task_prompt)

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(mask)
plt.title('Original mask Image')

plt.subplot(122)
plt.imshow(projected_img)
plt.title('Keypoint Projection')

plt.tight_layout()
plt.show()

print("Number of keypoints detected:", len(keypoints))
print("Keypoint positions:", keypoints)
