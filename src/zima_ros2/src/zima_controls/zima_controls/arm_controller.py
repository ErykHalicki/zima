import yaml
import numpy as np
from pathlib import Path
from sensor_msgs.msg import JointState
from zima_msgs.msg import ArmStateDelta
from zima_controls.ik_solver import IKSolver
from rclpy.node import Node

class ArmState:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, gripper=0.0, denorm_coeffs=None):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.gripper = gripper
        self.denorm_coeffs = denorm_coeffs

    def step(self, delta: ArmStateDelta):
        self.x += delta.translation.x * self.denorm_coeffs['translation']
        self.y += delta.translation.y * self.denorm_coeffs['translation']
        self.z += delta.translation.z * self.denorm_coeffs['translation']
        self.roll += delta.orientation.x * self.denorm_coeffs['rotation']
        self.pitch += delta.orientation.y * self.denorm_coeffs['rotation']
        self.yaw += delta.orientation.z * self.denorm_coeffs['rotation']
        self.gripper += delta.gripper_state * self.denorm_coeffs['gripper']

    def to_xyzrpy(self):
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

class ArmController(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.declare_parameter('initial_arm_state', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('denorm_config_path', '')
        self.declare_parameter('arm_structure_path', '')

        initial_state = self.get_parameter('initial_arm_state').value
        denorm_path = self.get_parameter('denorm_config_path').value
        arm_structure_path = self.get_parameter('arm_structure_path').value

        with open(denorm_path, 'r') as f:
            denorm_data = yaml.safe_load(f)
        denorm_coeffs = denorm_data['coefficients']

        self.arm_state = ArmState(*initial_state, gripper=0.0, denorm_coeffs=denorm_coeffs)

        self.ik_solver = IKSolver(arm_structure_path)

        self.delta_sub = self.create_subscription(
            ArmStateDelta,
            'arm_state_delta',
            self.delta_callback,
            10
        )

        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)

    def delta_callback(self, msg: ArmStateDelta):
        self.arm_state.step(msg)

        xyzrpy = self.arm_state.to_xyzrpy()
        joint_angles = list(self.ik_solver.solve(xyzrpy)) + [self.arm_state.gripper]

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']
        joint_state.position = joint_angles

        self.joint_state_pub.publish(joint_state)
