import os
os.environ["MUJOCO_GL"]="egl"
import matplotlib.pyplot as plt
import numpy as np
import time
from scp.utils.fast_ik_solver import IKSolver as slow_solver
from scp.utils.fast_ik_solver import IKSolver
import mujoco
import robosuite
import scp.utils.transform_utils as T
from scp.environment import ReKepMujocoEnv
from scp.utils.rekep_utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)
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
def print_ik_result(ik_result):
    """
    打印 IKResult 的信息。

    Args:
        ik_result (IKResult): IK 求解结果。
    """

    print("IK 求解结果:")
    print(f"  是否成功: {ik_result.success}")
    if ik_result.success:
        print(f"  关节位置: {ik_result.cspace_position}")
    print(f"  位置误差: {ik_result.position_error}")
    print(f"  姿态误差: {ik_result.orientation_error}")
    print(f"  迭代次数: {ik_result.num_descents}")

env = robosuite.make(
    env_name="NutAssembly",
    robots="Panda", 
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_segmentations=["instance", "class", "element"],
    camera_heights=512,
    camera_widths=512,
    camera_depths=True
)

global_config = get_config(config_path="./configs/config.yaml")
env = ReKepMujocoEnv(config=global_config['env'], env=env, verbose=True)
xml_path = '/root/ReKep_mujoco/aloha/vx300s_left.xml'

print('base_link:', global_config['env']['panda']['left_arm']['base_link'])

base_link = 'robot0_' + global_config['env']['panda']['left_arm']['base_link']
joint_names = ["robot0_" + name for name in global_config['env']['panda']['left_arm']['joint_names']]

ik_solver = IKSolver(
    env,
    base_link,
    joint_names,
)


#target_pose = np.array([-0.4,-0.04064606 , 0.4093267 ,  0.93970576  ,0.01095274,  0.10837643,-0.32417228])
target_pose = np.array([-0.1,-0.002,1.3,-0.00236351,0.99706794,-0.01982768,0.07387021])
# target_pose[3:7] = normalize_quaternion(target_pose[3:7])
# print(is_valid_quaternion(target_pose[3:7]))
# 进行20次求解并计算平均时间
num_iterations = 1
total_time = 0.0
successful_results = None
body_name = 'robot0_right_hand'
body_pos = np.array(env.physics.data.xpos[env.physics.model.body_name2id(body_name)])
body_quat = np.array(env.physics.data.xquat[env.physics.model.body_name2id(body_name)])  # 直接获取四元数
print(body_pos,body_quat)


for _ in range(num_iterations):
    start_time = time.time()
    result = ik_solver.solve(target_pose)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    successful_results = result
# solver = slow_solver(    env.physics,
#     global_config['env']['robot']['left_arm']['site_name'],
#     global_config['env']['robot']['left_arm']['joint_names'],)
# successful_results=solver.solve(target_pose)
# # 计算平均时间
average_time = total_time / num_iterations
print(f"IK 求解平均耗时: {average_time:.5f}s")

# #打印每次求解的结果
# for idx, result in enumerate(successful_results):
#     print(f"求解 {idx + 1}:")
#     print_ik_result(result)
print_ik_result(successful_results)
#打印平均时间:

# 如果最后一次求解成功，更新关节位置
# if successful_results.success:
#print ik
if True:
    env.physics.data.qpos[[env.physics.model.joint_name2id(name) for name in joint_names]] = successful_results.cspace_position[0:7]    #env.physics.named.data.qpos[left_arm_joint_names] = [0,0,0.3,0,0,0] 
    # 更新物理引擎状态，使关节角度生效
    env.physics.forward()
    # 获取 site 的姿态
    #print joint
    print('实际关节位置',env.physics.data.qpos)
    body_name = "robot0_right_hand"
    body_pos = np.array(env.physics.data.xpos[env.physics.model.body_name2id(body_name)])
    body_quat = np.array(env.physics.data.xquat[env.physics.model.body_name2id(body_name)])  # 直接获取四元数
    base_pos = np.array(env.physics.data.xpos[env.physics.model.body_name2id(base_link)])
    base_quat = np.array(env.physics.data.xquat[env.physics.model.body_name2id(base_link)])  # 直接获取四元数
    #print pos and quat relatived to base:
    pos_base, quat_base = pose_relative_to_base(body_pos, body_quat, base_pos, base_quat)
    print(f"Body '{body_name}' 相对于基座的位置: {pos_base}")
    print(f"Body '{body_name}' 相对于基座的四元数: {quat_base}")
    quat_base_xyzw = np.roll(quat_base,-1)
    print(quat_base_xyzw)
    mat_base = T.quat_to_rmat(quat_base_xyzw)
    euler = T.rmat_to_euler(mat_base)
    print(f"Body '{body_name}' 相对于基座的mat: {mat_base}")
    print(f"Body '{body_name}' 相对于基座的euler: {euler}")
    print(f"Body '{body_name}' 的姿态:")
    print(f"  位置: {body_pos}")
    print(f"  四元数: {body_quat}")
else:
    print("最后一次 IK 求解失败，无法设置关节角度")
