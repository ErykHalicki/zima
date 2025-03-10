from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
     return LaunchDescription([
        Node(
            package='hardware_integration',
            executable='serial_output',
        ),
        Node(
            package='arm_control',
            executable='IKSolver',
        )   ,
       ])
