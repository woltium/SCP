import torch
import numpy as np
import json
import os
import argparse
import robosuite as suite
from scp.environment import ScpMujocoEnv
from scp.keypoint_proposal import KeypointProposer
from scp.constraint_generation import ConstraintGenerator
from scp.subgoal_solver import SubgoalSolver
from scp.path_solver import PathSolver
from robosuite import load_controller_config
from scp.utils.fast_ik_solver import IKSolver
from scp.utils.visualizer import Visualizer
import scp.utils.transform_utils as T
import mimicgen
from scp.camera_utils import capture_current_point_cloud

from scp.utils.rekep_utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class Main:
    def __init__(self,task=None,visualize=False):
        self.global_config = get_config(config_path="./configs/config.yaml")
        self.description = self.global_config['task_descriptions'][task]['description']
        self.config = self.global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.camera_height = self.global_config['env']['camera'][0]['height']
        self.camera_width = self.global_config['env']['camera'][0]['width']
        self.visualize = visualize
        self.save_video = True
        if self.save_video:
            self.cameras = [self.config['vlm_camera'],self.config['obs_camera']]
        else:
             self.cameras = [self.config['vlm_camera']]
        self.task = task
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(self.global_config['keypoint_proposer'])
        self.constraint_generator = ConstraintGenerator(self.global_config['constraint_generator'])
        # initialize environment
        controller_name = "OSC_POSE"
        controller_config = load_controller_config(default_controller=controller_name)
        controller_config['control_delta'] = False
        # Create environment with camera segmentation enabled
        env = suite.make(
            env_name=self.global_config['task_descriptions'][task]['env'],
            robots="Panda", 
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self.cameras,
            camera_segmentations=["element",None],
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            camera_depths=[True,False],
            controller_configs=controller_config,
        )

        obs = env.reset()
        # Initialize the environment wrapper
        self.env = ScpMujocoEnv(env=env, verbose=True)
        # setup ik solver (for reachability cost)
        base_link = 'robot0_' + self.global_config['env']['panda']['left_arm']['base_link']
        joint_names = ["robot0_" + name for name in self.global_config['env']['panda']['left_arm']['joint_names']]
        ik_solver = IKSolver(
            self.env,
            base_link,
            joint_names,
        )
        # initialize solvers
        self.subgoal_solver = SubgoalSolver(self.global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(self.global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(self.global_config['visualizer'], self.env)


    def perform_task(self, scp_program_dir=None, disturbance_seq=None):
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']+'_image']
        points = capture_current_point_cloud(
            physics=self.env.physics,
            height=512,
            width=512,
            camera_name=self.config['vlm_camera']
        )
        mask = cam_obs[self.config['vlm_camera']+'_segmentation_element'][:, :, 0]

        # keypoint proposal and constraint generation

        if scp_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask,self.description)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            # if self.visualize:
            #     self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            scp_program_dir = self.constraint_generator.generate(projected_img, self.description, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(scp_program_dir, disturbance_seq)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, scp_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(scp_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(scp_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = None
            self.sdf_voxels = self.env.get_sdf_voxels()
            # if self.visualize:
            #     mesh = self.env.sdf_to_mesh(self.sdf_voxels,self.bounds_min,self.bounds_max,self.config['sdf_voxel_size'])
            #     self.visualizer.show_mesh_sep(mesh)
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info['num_stages']: 
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        if self.is_grasp_stage:
            preferred_grasp_pose = self.env.get_best_grasp_for_keypoint(self.program_info['grasp_keypoints'][self.stage - 1])
            if preferred_grasp_pose is not None:
                preferred_grasp_pose[3,:3] +=  preferred_grasp_pose[:3,:3]@ np.array([0, 0, self.config['grasp_depth']/2])
            else:
                preferred_grasp_pose = None
        else: 
            preferred_grasp_pose = None

        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            preferred_grasp_pose,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)

        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        # if self.visualize:
        #     self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose

    def _get_next_path(self, next_subgoal, from_scratch):
        path_constraints = self.constraint_fns[self.stage]['path']
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.last_gripper_action
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat_to_rmat(pregrasp_pose[3:]) @ np.array([0, 0, 2*self.config['grasp_depth']/3])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        print("staring grasp")
        self.env.execute_action(grasp_action, precise=True)
    
    def _execute_release_action(self):
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='StackThree', help='task to perform')
    parser.add_argument('--use_cached_query',action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    scp_program_dir = './vlm_query/'+args.task
    main = Main(task = args.task,visualize=args.visualize)
    main.perform_task(
                    scp_program_dir=scp_program_dir if args.use_cached_query else None)