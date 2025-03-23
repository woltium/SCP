from . import fast_ik
import numpy as np

class IKResult:
    """
    Represents the result of an IK solve attempt.
    """

    def __init__(self, success, cspace_position, position_error, orientation_error, num_descents):
        self.success = success
        self.cspace_position = cspace_position
        self.position_error = position_error
        self.orientation_error = orientation_error
        self.num_descents = num_descents


class IKSolver:
    def __init__(self, env,base, joint_names):

        self.env = env
        self.base_name = base
        self.joint_names = joint_names
    def solve(
        self,
        target_pose,
        max_iterations=100,
        initial_joint_pos=None,
        tol=0.01,
    ):
        target_pos = target_pose[:3]
        target_quat = target_pose[3:]

        result = fast_ik.ik(
            self.env.physics,
            base_name=self.base_name,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=self.joint_names, 
            tol=tol,
            max_steps=max_iterations,
        )

        return result