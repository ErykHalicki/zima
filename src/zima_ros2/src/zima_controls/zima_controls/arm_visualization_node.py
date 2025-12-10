#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

from zima_controls.kinematic_solver import KinematicSolver
from zima_controls.arm_visualizer import visualize_arm


class ArmVisualizationNode(Node):
    def __init__(self):
        super().__init__('arm_visualization_node')

        pkg_share = get_package_share_directory('zima_controls')
        arm_structure_path = os.path.join(pkg_share, 'arm_data', '4dof_arm_structure.yaml')
        arm_safety_path = os.path.join(pkg_share, 'arm_data', '4dof_arm_safety.yaml')

        self.solver = KinematicSolver(
            arm_structure_path,
            arm_safety_yaml_path=arm_safety_path,
            dimension_mask=[1,1,1,1,0,0]
        )

        self.subscription = self.create_subscription(
            JointState,
            '/current_joint_state',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Arm visualization node started')

    def joint_state_callback(self, msg):
        if len(msg.position)-1 != len(self.solver.revolute_links):
            self.get_logger().warn(f'Received joint state with {len(msg.position)} joints, expected {len(self.solver.revolute_links)}')
            return

        joint_angles = np.array(msg.position[:len(self.solver.revolute_links)])

        transforms = self.solver.forward(joint_angles)
        points = np.array([self.solver.transformation_matrix_to_translation(t) for t in transforms])

        visualize_arm([points], safety_boxes=self.solver.safety_boxes, display_time=0.5)


def main(args=None):
    rclpy.init(args=args)

    node = ArmVisualizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
