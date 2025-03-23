import numpy as np
import ikfastpy
import time
from scipy.spatial.transform import Rotation as R

# Initialize kinematics for UR5 robot arm
ur5_kin = ikfastpy.PyKinematics()
n_joints = ur5_kin.getDOF()

joint_angles = [0,0.1,0,0,0,0,0] # in radians
#joint_angles = [-1,0,0,0,0,0]
#joint_angles = [8.33737159e-16 ,2.07571758e-01 , 9.44228434e-01 , 1.23061703e-15,-1.15180019e+00, -7.63529446e-16]
# Test forward kinematics: get end effector pose from joint angles
print("\nTesting forward kinematics:\n")
print("Joint angles:")
print(joint_angles)
ee_pose = ur5_kin.forward(joint_angles)
ee_pose = np.asarray(ee_pose).reshape(3,4) # 3x4 rigid transformation matrix
print("\nEnd effector pose:")
print(ee_pose)
ee_rot = ee_pose[:3,:3]
r = R.from_matrix(ee_rot)

# 获取欧拉角 (顺序为 'xyz'，即先绕x轴旋转，再绕y轴，最后绕z轴)
euler_angles = r.as_euler('xyz')
print('euler_angles',euler_angles)
print("\n-----------------------------")

# Test inverse kinematics: get joint angles from end effector pose
print("\nTesting inverse kinematics:\n")
joint_configs = None
starttime = time.time()
#print time:
for i in range(1000):
    joint_configs = ur5_kin.inverse(ee_pose.reshape(-1).tolist())
print("Time taken: %.3f seconds"%(time.time()-starttime))
n_solutions = int(len(joint_configs)/n_joints)
print("%d solutions found:"%(n_solutions))
joint_configs = np.asarray(joint_configs).reshape(n_solutions,n_joints)
for joint_config in joint_configs:
    print(joint_config)

# Check cycle-consistency of forward and inverse kinematics
assert(np.any([np.sum(np.abs(joint_config-np.asarray(joint_angles))) < 1e-3 for joint_config in joint_configs]))
print("\nTest passed!")