import os
import numpy as np
import pinocchio as pin

from pinocchio.robot_wrapper import RobotWrapper
from gausstwin.utils.path_utils import get_franka_dir
from gausstwin.cfg.robot.fr3_v3_cfg import FR3_V3_LINK_LIST


class FrankaV3:
    def __init__(self):
        model_dir = get_franka_dir()
        urdf_path = os.path.join(model_dir, "fr3_v3.urdf")
        
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [model_dir])
        self.link_names = FR3_V3_LINK_LIST
        # self.link_indices = [3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        self.link_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        
        # set initial joint configuration
        self.q = pin.neutral(self.robot.model)  # type: ignore  # default neutral configuration
        self.q[:] = np.array([0.0478, -0.2635, 0.0355, -2.8931, -0.0169, 2.7762, 0.8408]) 


    def forward_kinematics(self, q: np.ndarray | None = None, dq: np.ndarray | None = None, h: float = 0.0):
        """Run forward kinematics for the robot."""
        if q is None:
            q = self.q
        if dq is None:
            dq = np.zeros_like(q)
        pin.forwardKinematics(self.robot.model, self.robot.data, q, dq) # type: ignore 
        pin.updateFramePlacements(self.robot.model, self.robot.data)  # type: ignore 
        
        # for i, frame in enumerate(self.robot.model.frames):
        #     if frame.type == pin.FrameType.BODY:
        #         pose = self.robot.data.oMf[i]
        #         print(i)
        #         print(f"{frame.name}: position={pose.translation}, rotation=\n{pose.rotation}")
        # for name, oMi in zip(self.robot.model.names, self.robot.data.oMi):
        #     print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
        
        # print(1/0)

        quaternions_wxyz_list = [np.roll(pin.Quaternion(self.robot.data.oMf[idx].rotation).coeffs(), 1)  # type: ignore
                            for idx in self.link_indices] 
        positions_list = [self.robot.data.oMf[idx].translation + np.array([0.0, 0.0, h]) for idx in self.link_indices]  # type: ignore

        quaternions_wxyz = np.array(quaternions_wxyz_list, dtype=np.float32)  # (N, 4)
        positions = np.array(positions_list, dtype=np.float32) # (N, 3)
        
        # for velocities
        linear_velocities = [pin.getFrameVelocity(self.robot.model, self.robot.data, idx, pin.ReferenceFrame.WORLD).linear  # type: ignore
                             for idx in self.link_indices]
        angular_velocities = [pin.getFrameVelocity(self.robot.model, self.robot.data, idx, pin.ReferenceFrame.WORLD).angular  # type: ignore
                              for idx in self.link_indices]
        linear_velocities = np.array(linear_velocities)
        angular_velocities = np.array(angular_velocities)
        
        return positions, quaternions_wxyz, linear_velocities, angular_velocities
    
    
    def forward_kinematics_interpolation(self, q0: np.ndarray | None = None, q1: np.ndarray | None=None, dq: np.ndarray | None = None, h: float = 0.0, N: int = 10):
        """Run forward kinematics for the robot with interpolation."""
        if q0 is None:
            q0 = self.q
        if q1 is None:
            q1 = self.q
        
        # Interpolate N steps between q0 and q1
        positions_list = []
        quaternions_wxyz_list = []
        for i in range(N):
            u = i * 1.0 / (N - 1)
            # qi = self.robot.model.interpolate(q0, q1, u)
            qi = pin.interpolate(self.robot.model, q0, q1, u) # type: ignore

            positions, quaternions_wxyz, linear_vel, angular_vel = self.forward_kinematics(q=qi, dq=dq, h=h)

            positions_list.append(positions)
            quaternions_wxyz_list.append(quaternions_wxyz)
            
        return positions_list, quaternions_wxyz_list, linear_vel, angular_vel
    