import time
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import trimesh
import open3d as o3d
import imageio
from skimage import measure
from scp.utils.visualizer import Visualizer
import mujoco

# Import mujoco from dm_control
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_real_depth_map
#import transform_utils as T
from scp.utils.rekep_utils import (
    bcolors,
    get_config,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)

class ScpMujocoEnv:                                                                                      
    def __init__(self,env,verbose=False,grapsp_adviser='anygrasp'):
        # Load configuration and initialize environment parameters
        config_path=os.path.join(os.path.dirname(__file__), "configs/config.yaml")
        global_config = get_config(config_path=config_path)
        self.vlm_camera = global_config['main']['vlm_camera']
        self.obs_camera = global_config['main']['obs_camera']
        self.video_cache = []
        self.verbose = verbose
        self.interpolate_pos_step_size = global_config['main']['interpolate_pos_step_size']
        self.interpolate_rot_step_size = global_config['main']['interpolate_rot_step_size']
        self.camera_height = global_config['env']['camera'][0]['height']
        self.camera_width = global_config['env']['camera'][0]['width']
        self.video_cache_size = global_config['env']['video_cache_size']
        self.resolution = global_config['main']['sdf_voxel_size']
        
        # Initialize MuJoCo environment
        self.env = env
        self.bounds_min = np.array(global_config['main']['bounds_min'])
        self.bounds_max = np.array(global_config['main']['bounds_max'])
        self.last_gripper_action = self.get_gripper_open_action()
        # Keypoint registry
        self._keypoint_registry = None
        self._keypoint2object = None
        self.reset_joint_pos = self.env.robots[0]._joint_positions
        self.visualize = True
        self.step_counter = 0 
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)
        self.left_in_hand_geom_ids = set()
        self.right_in_hand_geom_ids = set()
        self.exclude_list = ["frame", "vx300", "floor", ]
        # if grapsp_adviser == 'anygrasp':
        #     from scp.anygrasp.get_grasp import GraspDetector
        #     camera_instrics = self.camera_instrics(self.camera_width,self.camera_height,self.vlm_camera)
        #     self.grasp_detector = GraspDetector(camera_intrinsics = camera_instrics)

    def init_plt(self):
        ts = self.env.step(None)
        self.plt_tele = self.ax_tele.imshow(ts.observation['images']['rgb']['teleoperator_pov'])
        self.plt_worm = self.ax_worm.imshow(ts.observation['images']['rgb']['vlm_cam'])
    
    # Core Environment Operations
    @property
    def physics(self):
        return self.env.sim

    def reset(self):
        self.env.reset()

    def render(self):
        self.env.render()

    def sleep(self, seconds):
        time.sleep(seconds)
        
    # Geometric Processing Utilities
    def sdf_to_mesh(self, sdf_voxels):
        verts, faces, _, _ = measure.marching_cubes(sdf_voxels, level=0, spacing=(self.resolution, self.resolution, self.resolution))
        verts += self.bounds_min
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh

    def mujoco_mesh_to_trimesh(self, geom_id):
        mesh_id = self.physics.model.geom_dataid[geom_id]
        vert_start = self.physics.model.mesh_vertadr[mesh_id]
        vert_count = self.physics.model.mesh_vertnum[mesh_id]
        vertices = self.physics.model.mesh_vert[vert_start:vert_start + vert_count].reshape(-1, 3)
        
        face_start = self.physics.model.mesh_faceadr[mesh_id]
        face_count = self.physics.model.mesh_facenum[mesh_id]
        faces = self.physics.model.mesh_face[face_start:face_start + face_count].reshape(-1, 3)
        trimesh_object = trimesh.Trimesh(vertices=vertices, faces=faces)

        return trimesh_object
    
    #Sensor Data Processing
    def camera_instrics(self,width,height,camera_name):
        camera_id = self.physics.model.camera_name2id(camera_name)
        fov = self.physics.model.cam_fovy[camera_id]
        theta = np.deg2rad(fov)
        fx = width / 2 / np.tan(theta / 2)
        fy = height / 2 / np.tan(theta / 2)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return fx, fy, cx, cy

    def get_sdf_voxels(self, exclude_robot=True, exclude_obj_in_hand=True):
        exclude_body_ids = self.get_contact_body_ids_optimized('gripper')
        exclude_list = ["robot","gripper"]
        trimesh_objects = []
        for i in range(self.physics.model.ngeom):

            geom_name = self.physics.model.geom_id2name(i)
            geom_body_id = self.physics.model.geom_bodyid[i]
            
            if geom_body_id in exclude_body_ids:
                continue
            if geom_name and any(exclude_str in geom_name for exclude_str in exclude_list):
                continue
            geom_type = self.physics.model.geom_type[i]
            geom_size = self.physics.model.geom_size[i]

            # Get geom pose
            geom_pos = self.physics.data.geom_xpos[i]
            geom_quat = self.physics.data.geom_xmat[i].reshape(3, 3)
            geom_quat = trimesh.transformations.quaternion_from_matrix(
                np.concatenate([geom_quat, np.array([geom_pos]).T], axis=1)
            )
            # Create trimesh mesh based on geom type
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                trimesh_object = trimesh.creation.box(extents=geom_size * 2)  # Mujoco size is half-extents
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                trimesh_object = trimesh.creation.icosphere(radius=geom_size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                trimesh_object = trimesh.creation.cylinder(
                    radius=geom_size[0], height=geom_size[1] * 2
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                trimesh_object = self.mujoco_mesh_to_trimesh(i)
                geom_id = self.physics.model.geom_bodyid[i]
            else:
                #print(f"Warning: Unsupported geom type {geom_type} ")
                continue

            # Apply geom pose
            trimesh_object.apply_transform(
                trimesh.transformations.translation_matrix(geom_pos)
                @ trimesh.transformations.quaternion_matrix(geom_quat)
            )
            trimesh_objects.append(trimesh_object)

        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        scene = o3d.t.geometry.RaycastingScene()
        vertex_positions = o3d.core.Tensor(scene_mesh.vertices, dtype=o3d.core.Dtype.Float32)
        triangle_indices = o3d.core.Tensor(scene_mesh.faces, dtype=o3d.core.Dtype.UInt32)
        _ = scene.add_triangles(vertex_positions, triangle_indices)

        shape = np.ceil((self.bounds_max - self.bounds_min) / self.resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape
        grid = np.mgrid[self.bounds_min[0]:self.bounds_max[0]:steps[0],
                        self.bounds_min[1]:self.bounds_max[1]:steps[1],
                        self.bounds_min[2]:self.bounds_max[2]:steps[2]]
        grid = grid.reshape(3, -1).T

        sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
        sdf_voxels = -sdf_voxels.cpu().numpy()
        sdf_voxels = sdf_voxels.reshape(shape)

        #self.visualize_sdf(sdf_voxels,self.bounds_min,self.bounds_max)
        return sdf_voxels
    
    def visualize_sdf_interior(self,sdf_voxels, bounds_min, bounds_max):
        # Get the coordinates of the interior points (points where the SDF is negative)
        interior_points = np.where(sdf_voxels >0)
        
        # convert back to actual coordinate
        shape = sdf_voxels.shape
        steps = (bounds_max - bounds_min) / shape
        x = bounds_min[0] + interior_points[0] * steps[0]
        y = bounds_min[1] + interior_points[1] * steps[1] 
        z = bounds_min[2] + interior_points[2] * steps[2]

        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=sdf_voxels[interior_points], 
                            cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Interior Points (SDF < 0)')
        plt.show()
        
    def get_contact_body_ids_optimized(self, geom_name):
        target_geom_ids = [
            i for i in range(self.physics.model.ngeom)
            if self.physics.model.geom_id2name(i) and geom_name in self.physics.model.geom_id2name(i)
        ]
        contact_body_ids = set()  # Use a set to avoid duplicate geom IDs
        for contact in self.physics.data.contact:
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            if geom1_id in target_geom_ids and geom2_id not in target_geom_ids:
                contact_body_ids.add(self.physics.model.geom_bodyid[geom2_id])
            elif geom2_id in target_geom_ids and geom1_id not in target_geom_ids:
                contact_body_ids.add(self.physics.model.geom_bodyid[geom1_id])

        return list(contact_body_ids)  # Convert the set to a list
    
    def get_contact_geom_ids_optimized(self, geom_name):
        target_geom_ids = [
            i for i in range(self.physics.model.ngeom)
            if self.physics.model.geom_id2name(i) and geom_name in self.physics.model.geom_id2name(i)
        ]
        contact_geom_ids = set()  # Use a set to avoid duplicate geom IDs
        
        for contact in self.physics.data.contact:
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            if geom1_id in target_geom_ids and geom2_id not in target_geom_ids:
                contact_geom_ids.add(geom2_id)
            elif geom2_id in target_geom_ids and geom1_id not in target_geom_ids:
                contact_geom_ids.add(geom1_id)

        return list(contact_geom_ids)  # Convert the set to a list
    
    def register_keypoints(self, keypoints):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.keypoints = keypoints
        self._keypoint_registry = dict()
        self._keypoint2object = dict()
        exclude_names = ['frame', 'vx300', 'floor', 'table'] # Replace with MuJoCo specific names

        for idx, keypoint in enumerate(keypoints):
            closest_distance = np.inf
            closest_geom_id = -1
            closest_point = None

            for i in range(self.physics.model.ngeom):
                geom_name = self.physics.model.geom_id2name(i)
                if geom_name == None or any(name in geom_name.lower() for name in exclude_names) :
                    continue

                # Get geom type and size
                geom_type = self.physics.model.geom_type[i]

                # Get geom pose
                geom_pos = self.physics.data.geom_xpos[i]
                geom_quat = self.physics.data.geom_xmat[i].reshape(3, 3)
                geom_quat = trimesh.transformations.quaternion_from_matrix(
                    np.concatenate([geom_quat, np.array([geom_pos]).T], axis=1)
                )

                # Create trimesh mesh based on geom type
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    trimesh_object = trimesh.creation.box(extents=self.physics.model.geom_size[i] * 2)
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    trimesh_object = trimesh.creation.icosphere(radius=self.physics.model.geom_size[i][0])
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    trimesh_object = trimesh.creation.cylinder(
                        radius=self.physics.model.geom_size[i][0], height=self.physics.model.geom_size[i][1] * 2
                    )
                elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                    trimesh_object = self.mujoco_mesh_to_trimesh(i)
                else:
                    continue

                # Apply geom pose
                trimesh_object.apply_transform(
                    trimesh.transformations.translation_matrix(geom_pos)
                    @ trimesh.transformations.quaternion_matrix(geom_quat)
                )

                points_transformed = trimesh_object.sample(1000)

                # find closest point
                dists = np.linalg.norm(points_transformed - keypoint, axis=1)
                point = points_transformed[np.argmin(dists)]
                distance = np.linalg.norm(point - keypoint)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_geom_id = i
                    closest_point = point

            geom_pos = self.physics.data.geom_xpos[closest_geom_id]
            geom_rot = self.physics.data.geom_xmat[closest_geom_id].reshape(3,3)
            # Build transform matrix from world to geom local frame
            geom_pose = np.eye(4)
            geom_pose[:3, :3] = geom_rot
            geom_pose[:3, 3] = geom_pos
            geom_pose_inv = np.linalg.inv(geom_pose)
            
            # Transform keypoint to geom's local frame
            keypoint_local = geom_pose_inv.dot(np.append(closest_point, 1))[:3]
            
            # Store local coordinates and geom id
            self._keypoint_registry[idx] = (closest_geom_id, keypoint_local)
            self._keypoint2object[idx] = closest_geom_id
            
    def register_keypoint_from_local(self,keypoint_registry):
        self._keypoint_registry = keypoint_registry
        for idx, (geom_id, local_coords) in keypoint_registry.items():
            self._keypoint2object[idx] = geom_id  

    def get_keypoint_positions(self):
        keypoint_positions = []
        for idx, (geom_id, keypoint_local) in self._keypoint_registry.items():
            # Get current geom pose
            curr_pos = self.physics.data.geom_xpos[geom_id]
            curr_rot = self.physics.data.geom_xmat[geom_id].reshape(3,3)
            
            # Build current transform matrix
            curr_pose = np.eye(4)
            curr_pose[:3, :3] = curr_rot
            curr_pose[:3, 3] = curr_pos
            
            # Transform local coordinates back to world frame
            keypoint_world = curr_pose.dot(np.append(keypoint_local, 1))[:3]
            keypoint_positions.append(keypoint_world)
            
        return np.array(keypoint_positions)


    def get_object_by_keypoint(self, keypoint_idx):
        assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint2object[keypoint_idx]
    
    def get_best_grasp_for_keypoint(self, keypoint_idx):
        """
        Get the best grasp pose for the object associated with the given keypoint by trying multiple cameras.
        
        Args:
            keypoint_idx (int): Index of the keypoint associated with target object
            
        Returns:
            best_grasp: Best grasp pose found from any camera, or None if no valid grasps
        """
        # obs = self.env.observation_spec()
        # cameras = [self.vlm_camera]
        
        # for camera in cameras:
        #     rgb = obs[camera+'_image'] 
        #     depth = obs[camera+'_depth'][:,:,0]
        #     real_depth = get_real_depth_map(self.physics, depth)
        #     seg_mask = obs[camera+'_segmentation_element'][:, :, 0]

        #     target_geom_id = self.get_object_by_keypoint(keypoint_idx)
        #     instance_name = self.env.model.geom_ids_to_instances[target_geom_id]
        #     instance_id = self.env.model.instances_to_ids[instance_name]
        #     print(f"Target object instance name: {instance_name}, instance id: {instance_id}")
            
        #     object_mask = np.zeros_like(seg_mask, dtype=bool)
        #     for geom_id in instance_id['geom']:
        #         object_mask |= (seg_mask == geom_id)

        #     y_indices, x_indices = np.where(object_mask)
        #     if len(y_indices) > 0 and len(x_indices) > 0:
        #         x_min, x_max = np.min(x_indices), np.max(x_indices)
        #         y_min, y_max = np.min(y_indices), np.max(y_indices)
        #         print(f"Object bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")    

        #     grasps = self.grasp_detector.get_grasps(rgb, real_depth, object_mask)
            
        #     if grasps is not None:
        #         best_grasp = grasps[0]
        #         best_grasp = self.compute_final_grasp_pose(best_grasp.translation, best_grasp.rotation_matrix, camera)
        #         return best_grasp
                
        return None
    
    def compute_final_grasp_pose(self,translation, rotation, camera_name):
        # Convert point coordinates
        camera_extr = get_camera_extrinsic_matrix(self.physics,camera_name)
        point = np.array([translation[0], translation[1], translation[2]])
        
        # Transformation matrix from camera to model coordinate system
        rot_matrix = np.array([[1, 0, 0],
                            [0, 1, 0], 
                            [0, 0, 1]])
        
        # Calculate rotation
        rotation_cam = rot_matrix @ rotation @ rot_matrix
        
        # Build transformation matrix
        dest_frame = np.eye(4)
        dest_frame[:3,:3] = rotation_cam
        dest_frame[:3,3] = point
        
        # 应用相机位姿变换
        transformed_frame = camera_extr @ dest_frame
        print("transformed_frame: ", transformed_frame)
        # 应用夹爪姿态变换
        gripper_rot = np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0,-1, 0]])
        transformed_frame[:3,:3] = transformed_frame[:3,:3] @ gripper_rot

        
        return transformed_frame

    def get_collision_points(self, noise=True):
        """
        Get the points of the gripper and any object in hand.
        """
        # add gripper collision points
        
        collision_points = []
        geom_ids = [
            i for i in range(self.physics.model.ngeom)
            if  self.physics.model.geom_id2name(i) and any(keyword in self.physics.model.geom_id2name(i) for keyword in ['finger'])
        ]
        # Add object in hand
        geom_ids.extend(self.get_contact_geom_ids_optimized('gripper'))
        #print('collision_geom_name',[self.physics.model.geom_id2name(i) for i in geom_ids])        
        for i in geom_ids:
            # Get geom type and size
            geom_type = self.physics.model.geom_type[i]
            # Get geom pose
            geom_pos = self.physics.data.geom_xpos[i]
            geom_quat = self.physics.data.geom_xmat[i].reshape(3, 3)
            geom_quat = trimesh.transformations.quaternion_from_matrix(
                np.concatenate([geom_quat, np.array([geom_pos]).T], axis=1)
            )

            # Create trimesh mesh based on geom type
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                trimesh_object = trimesh.creation.box(extents=self.physics.model.geom_size[i] * 2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                trimesh_object = trimesh.creation.icosphere(radius=self.physics.model.geom_size[i][0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                trimesh_object = trimesh.creation.cylinder(
                    radius=self.physics.model.geom_size[i][0], height=self.physics.model.geom_size[i][1] * 2
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                trimesh_object = self.mujoco_mesh_to_trimesh(i)
            else:
                continue

            # Apply geom pose
            trimesh_object.apply_transform(
                trimesh.transformations.translation_matrix(geom_pos)
                @ trimesh.transformations.quaternion_matrix(geom_quat)
            )

            points_transformed = trimesh_object.sample(1000)

            # add to collision points
            collision_points.append(points_transformed)

        # Assuming object in hand is attached to the gripper, its collision points will be included above

        collision_points = np.concatenate(collision_points, axis=0)
        return collision_points
    
    # Robot Control and Grasping
    def execute_action(
            self,
            action,
            precise=True,
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qw, qx, qy, qz, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            if precise:
                pos_threshold = 0.01
                rot_threshold = 3.0
            else:
                pos_threshold = 0.10
                rot_threshold = 5.0
            action = np.array(action).copy()
            assert action.shape == (8,)
            target_pose = action[:7]
            gripper_action = action[7]

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            current_pos = self.get_ee_pos()
            current_quat = self.get_ee_quat()
            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pos - target_pose[:3])
            rot_diff = angle_between_quats(current_quat, target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            intermediate_pos_threshold = 0.05
            intermediate_rot_threshold = 5.0
            for pose in pose_seq[:-1]:
                self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
            # move to the final pose with required precision
            pose = pose_seq[-1]
            self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action
            # ======================================
            if gripper_action == self.get_gripper_open_action():
                self.open_gripper()
            elif gripper_action == self.get_gripper_close_action():
                self.close_gripper()
            elif gripper_action == self.get_gripper_null_action():
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            
            return pos_error, rot_error
        
    def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=20):
        pos_errors = []
        rot_errors = []
        count = 0
        reached = False
        while count < max_steps:
            pos_error, rot_error = self.compute_target_delta_ee(target_pose_world)
            if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
                reached = True
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                print("reached")
                break
            action = np.zeros(8)
            action[:7] = target_pose_world
            action[7] = self.last_gripper_action
            _ = self._step(action=action)
            count += 1
        if count == max_steps:
            print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(4)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

    def _step(self, action=None):
        #transform quat to axisangle
        axisangle = T.quat2axisangle(action[3:7])
        action = np.concatenate([action[0:3], axisangle, action[7:]])
        obs, reward, done, info = self.env.step(action)
        self.render()
        imgs = obs[self.obs_camera+'_image']
        #rgb = imgs['rgb']['teleoperator_pov']
        # if self.visualize:
        #     self.plt_tele.set_data(imgs['rgb']['teleoperator_pov'])
        #     self.plt_worm.set_data(imgs['rgb']['vlm_cam'])
        #     plt.pause(0.02)
        if len(self.video_cache) < self.video_cache_size:
            self.video_cache.append(imgs)
        else:
            self.video_cache.pop(0)
            self.video_cache.append(imgs)
        self.step_counter += 1
        
    def is_grasping(self, candidate_obj_id):
        candidate_body_id = self.physics.model.geom_bodyid[candidate_obj_id] #获取物体的body id
        contact_body_ids = self.get_contact_body_ids_optimized('finger')
        return candidate_body_id in contact_body_ids

    def _get_body_pose_by_name(self, body_name):
        body_id = self.physics.model.name2id(body_name, 'body')
        pos = self.physics.data.xpos[body_id].copy()
        quat_wxyz = self.physics.data.xquat[body_id].copy()
        return np.concatenate([pos, quat_wxyz])
    
    def get_ee_pose(self):
        return np.concatenate([self.get_ee_pos(), self.get_ee_quat()])

    def get_ee_pos(self):
        return self.env._eef_xpos

    def get_ee_quat(self):
        return self.env._eef_xquat
    
    def get_cam_obs(self):
        obs = self.env.observation_spec()
        return obs
    
    def open_gripper(self):
        """
        Opens the gripper while maintaining the current end-effector pose.
        """
        # Get the current end-effector pose
        current_pose = self.get_ee_pose()
        
        # Create a new action with the open gripper action and the current pose
        action = np.concatenate([current_pose[:7],[self.get_gripper_open_action()]])
        # Execute the action
        for _ in range(30):
            self._step(action)
        self.last_gripper_action = self.get_gripper_open_action()
        
    def close_gripper(self):
        """
        Opens the gripper while maintaining the current end-effector pose.
        """
        # Get the current end-effector pose
        current_pose = self.get_ee_pose()

        # Create a new action with the open gripper action and the current pose
        action = np.concatenate([current_pose[:7], [self.get_gripper_close_action()]])
        # Execute the action
        for _ in range(30):
            self._step(action)
        self.last_gripper_action = self.get_gripper_close_action()
    
    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def compute_target_delta_ee(self, target_pose):
        target_pos, target_wxyz = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_wxyz = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_wxyz, target_wxyz)
        
        return pos_diff, rot_diff
    
    def save_video(self, save_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(save_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            flipped_frame = np.flipud(rgb)
            video_writer.append_data(flipped_frame)
        video_writer.close()
        print("step_counter:",self.step_counter)
        return save_path
    
    def visualize_sdf(self, sdf_voxels, bounds_min, bounds_max):
        from mpl_toolkits.mplot3d import Axes3D
        from skimage import measure
        
        # Extract vertices and faces using marching cubes
        verts, faces, _, _ = measure.marching_cubes(sdf_voxels, level=0, spacing=(self.resolution, self.resolution, self.resolution))
        verts += bounds_min
        
        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface triangles
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces, cmap='viridis')
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SDF Visualization')
        
        # Show plot
        plt.show()
        