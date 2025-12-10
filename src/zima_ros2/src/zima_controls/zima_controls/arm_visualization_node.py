#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from ament_index_python.packages import get_package_share_directory

from zima_controls.kinematic_solver import KinematicSolver
from zima_controls.arm_visualizer import _draw_safety_boxes


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

        self.current_joint_angles = None
        self.current_points = None

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, marker='o', markersize=6)
        self.base_point, = self.ax.plot([], [], [], 'go', markersize=10, label='Base')
        self.end_effector_point, = self.ax.plot([], [], [], 'ro', markersize=10, label='End Effector')

        self.setup_plot()

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)

        self.get_logger().info('Arm visualization node started')

    def setup_plot(self):
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Arm Visualization')
        self.ax.legend()

        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.8])

        if hasattr(self.solver, 'safety_boxes') and self.solver.safety_boxes:
            _draw_safety_boxes(self.ax, self.solver.safety_boxes)

    def joint_state_callback(self, msg):
        if len(msg.position) != len(self.solver.revolute_links):
            self.get_logger().warn(f'Received joint state with {len(msg.position)} joints, expected {len(self.solver.revolute_links)}')
            return

        joint_angles_degrees = np.array(msg.position[:len(self.solver.revolute_links)]) - 90.0
        joint_angles_radians = np.radians(joint_angles_degrees)

        self.current_joint_angles = joint_angles_radians

        transforms = self.solver.forward(joint_angles_radians)
        self.current_points = np.array([t[:3, 3] for t in transforms])

    def update_plot(self, frame):
        if self.current_points is not None:
            points = self.current_points

            self.line.set_data(points[:, 0], points[:, 1])
            self.line.set_3d_properties(points[:, 2])

            self.base_point.set_data([points[0, 0]], [points[0, 1]])
            self.base_point.set_3d_properties([points[0, 2]])

            self.end_effector_point.set_data([points[-1, 0]], [points[-1, 1]])
            self.end_effector_point.set_3d_properties([points[-1, 2]])

        return self.line, self.base_point, self.end_effector_point


def main(args=None):
    rclpy.init(args=args)

    node = ArmVisualizationNode()

    try:
        plt.show()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
