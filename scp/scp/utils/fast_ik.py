import numpy as np
import ikfastpy
import mujoco

def pose_relative_to_base(target_pos, target_quat, base_pos, base_quat):
    # Convert input arrays to numpy arrays for easier manipulation
    target_pos = np.array(target_pos)
    base_pos = np.array(base_pos)

    # Calculate the position of the target in the base frame
    target_pos_base = target_pos - base_pos

    # Calculate the relative quaternion
    neg_base_quat = np.empty(4)
    mujoco.mju_negQuat(neg_base_quat, base_quat)  # Get the negative of the base quaternion
    target_quat_base = np.empty(4)
    mujoco.mju_mulQuat(target_quat_base, target_quat, neg_base_quat)  # Target relative to base

    return target_pos_base, target_quat_base

class IKResult:
    def __init__(self, success, cspace_position , position_error, orientation_error, num_descents):
        self.success = success
        self.cspace_position  = cspace_position 
        self.position_error = position_error
        self.orientation_error = orientation_error
        self.num_descents = num_descents
        
def ik(physics, base_name, target_pos, target_quat, joint_names, tol, max_steps):
    """
    Fast inverse kinematics using ikfastpy and mjbindings.
    """

    robot = ikfastpy.PyKinematics()
    n_joints = robot.getDOF()

    # Get current joint angles
    current_joint_angles = np.array([physics.data.qpos[physics.model.joint_name2id(joint_name)] for joint_name in joint_names])
    current_joint_angles = np.squeeze(current_joint_angles)

    # Get site pose in world frame
    # Calculate target pose in robot base frame using mju_mulPose
    base_xpos = physics.data.body_xpos[physics.model.body_name2id(base_name)]
    #print('base_xpos:', base_xpos)
    base_xmat = physics.data.body_xmat[physics.model.body_name2id(base_name)]
    
    base_xquat = np.empty(4)
    mujoco.mju_mat2Quat(base_xquat, base_xmat)
    #print base_xpos,base_xquat
    target_pos_base,target_quat_base = pose_relative_to_base(target_pos, target_quat, base_xpos, base_xquat)


    target_mat_base = np.eye(4)
    res = np.empty(9)
    mujoco.mju_quat2Mat(res, target_quat_base)
    target_mat_base[:3, :3] = res.reshape(3, 3)  # 将结果转换为 3x3 矩阵并复制
    target_mat_base[:3, 3] = target_pos_base
    ee_pose = target_mat_base[:3, :]
    ee_pose = np.asarray(ee_pose).reshape(3,4)
    
    # Solve inverse kinematics
    joint_configs = robot.inverse(ee_pose.reshape(-1).tolist())
    n_solutions = int(len(joint_configs) / n_joints)
    joint_configs = np.asarray(joint_configs).reshape(n_solutions, n_joints)


    if n_solutions > 0:
        diffs = np.sum((joint_configs - current_joint_angles) ** 2, axis=1)
        best_solution_idx = np.argmin(diffs)
        best_solution = joint_configs[best_solution_idx]

        # Calculate position error
        ee_pose_solved = robot.forward(best_solution.tolist())
        ee_pose_solved = np.asarray(ee_pose_solved).reshape(3, 4)
        position_error = np.linalg.norm(ee_pose_solved[:3, 3] - target_pos_base)

        solved_rot_mat = ee_pose_solved[:3, :3]  # Get rotation matrix from solved pose
        solved_quat = np.empty(4)
        mujoco.mju_mat2Quat(solved_quat, solved_rot_mat.reshape(-1))  # Convert to quaternion

        # Calculate orientation error 
        neg_solved_quat = np.empty(4)
        mujoco.mju_negQuat(neg_solved_quat, solved_quat)
        err_rot_quat = np.empty(4)
        mujoco.mju_mulQuat(err_rot_quat, target_quat_base, neg_solved_quat)  # Compare with target in base frame
        err_rot = np.empty(3)
        mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
        orientation_error = np.linalg.norm(err_rot)

        # Check if solution is within tolerance
        success = position_error < tol and orientation_error < tol

        return IKResult(
            success, best_solution, position_error, orientation_error, 1
        )
    else:
        return IKResult(False, None, float("inf"), float("inf"), 1)
