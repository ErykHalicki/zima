import yaml
import numpy as np
from sensor_msgs.msg import JointState
from zima_msgs.msg import ArmStateDelta
from zima_controls.kinematic_solver import KinematicSolver
from rclpy.node import Node
from std_msgs.msg import Empty
import rclpy
import copy

class ArmState:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, gripper=0.0, denorm_coeffs=None, limits=None):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.gripper = gripper
        self.denorm_coeffs = denorm_coeffs
        self.limits = limits

    def step(self, delta: ArmStateDelta):
        self.x += max(-1, min(1, delta.translation.x)) * self.denorm_coeffs['translation']
        self.y += max(-1, min(1, delta.translation.y)) * self.denorm_coeffs['translation']
        self.z += max(-1, min(1, delta.translation.z)) * self.denorm_coeffs['translation']
        self.roll += max(-1, min(1, delta.orientation.x)) * self.denorm_coeffs['rotation']
        self.pitch += max(-1, min(1, delta.orientation.y)) * self.denorm_coeffs['rotation']
        self.yaw += max(-1, min(1, delta.orientation.z)) * self.denorm_coeffs['rotation']
        self.gripper += max(-1, min(1, delta.gripper)) * self.denorm_coeffs['gripper']

        if self.limits:
            self.x = max(self.limits['x']['min'], min(self.limits['x']['max'], self.x))
            self.y = max(self.limits['y']['min'], min(self.limits['y']['max'], self.y))
            self.z = max(self.limits['z']['min'], min(self.limits['z']['max'], self.z))
            self.roll = max(self.limits['roll']['min'], min(self.limits['roll']['max'], self.roll))
            self.pitch = max(self.limits['pitch']['min'], min(self.limits['pitch']['max'], self.pitch))
            self.yaw = max(self.limits['yaw']['min'], min(self.limits['yaw']['max'], self.yaw))
            self.gripper = max(self.limits['gripper']['min'], min(self.limits['gripper']['max'], self.gripper))

    def to_xyzrpy(self):
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

class ArmController(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.declare_parameter('initial_arm_state', [0.2, 0.0, 0.1, 0.0, 0.0, 0.0])
        self.declare_parameter('denorm_config_path', '')
        self.declare_parameter('arm_structure_path', '')
        self.declare_parameter('arm_safety_path', '')
        self.declare_parameter('max_error_thresh', 0.05)

        initial_state = self.get_parameter('initial_arm_state').value
        denorm_path = self.get_parameter('denorm_config_path').value
        arm_structure_path = self.get_parameter('arm_structure_path').value
        arm_safety_path = self.get_parameter('arm_safety_path').value
        if arm_safety_path == '':
            arm_safety_path = None
        self.max_error_thresh = self.get_parameter('max_error_thresh').value
        self.last_joint_state = None

        with open(denorm_path, 'r') as f:
            denorm_data = yaml.safe_load(f)
        denorm_coeffs = denorm_data['coefficients']
        limits = denorm_data.get('limits', None)

        self.initial_state = initial_state
        self.denorm_coeffs = denorm_coeffs
        self.limits = limits

        self.arm_state = ArmState(*initial_state, gripper=0.0, denorm_coeffs=denorm_coeffs, limits=limits)

        self.kinematic_solver = KinematicSolver(arm_structure_path, arm_safety_yaml_path=arm_safety_path, dimension_mask=[1,1,1,1,0,0])

        self.delta_sub = self.create_subscription(
            ArmStateDelta,
            '/arm_state_delta',
            self.delta_callback,
            10
        )

        self.reset_sub = self.create_subscription(
            Empty,
            '/reset_arm',
            self.reset_callback,
            10
        )

        self.joint_state_pub = self.create_publisher(JointState, '/goal_joint_state', 10)

    def delta_callback(self, msg: ArmStateDelta):
        arm_backup = copy.copy(self.arm_state)
        self.arm_state.step(msg)

        xyzrpy = self.arm_state.to_xyzrpy()
        joint_states, error, iterations = self.kinematic_solver.solve(xyzrpy[:3], xyzrpy[3:], self.last_joint_state)
        safe, violating_joint_name = self.kinematic_solver.is_joint_state_safe(joint_states)
        if(error >= self.max_error_thresh):
            self.get_logger().warn(f"IK Error {error} > {self.max_error_thresh} treshold. Not publishing joint state", throttle_duration_sec=2.0)
            self.get_logger().warn(f"Arm state: {xyzrpy}", throttle_duration_sec=2.0)
            self.arm_state = arm_backup
            return
        elif not safe:
            self.get_logger().warn(f"Unsafe IK solution. {violating_joint_name} in unsafe position. Not publishing joint state.", throttle_duration_sec=2.0)
            self.get_logger().warn(f"Arm state: {xyzrpy}", throttle_duration_sec=2.0)
            self.arm_state = arm_backup
            return

        self.last_joint_state = joint_states
        total_joint_angles = list(joint_states) + [self.arm_state.gripper]

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['base', 'shoulder', 'elbow', 'wrist', 'gripper']
        joint_state.position = np.degrees(total_joint_angles) + 90

        self.joint_state_pub.publish(joint_state)

    def reset_callback(self, msg: Empty):
        self.arm_state = ArmState(*self.initial_state, gripper=0.0, denorm_coeffs=self.denorm_coeffs, limits=self.limits)
        self.last_joint_state = None
        self.get_logger().info('Arm state reset to initial pose')

def main(args=None):
    rclpy.init(args=args)
    node = ArmController('arm_controller')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
