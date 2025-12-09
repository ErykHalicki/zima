from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    zima_controls_dir = get_package_share_directory('zima_controls')

    arm_data_path = os.path.join(zima_controls_dir, 'arm_data')


    arm_control_node = Node(
            package='zima_controls',
            executable='arm_controller',
            name='arm_controller',
            parameters=[{
                'arm_structure_path': os.path.join(arm_data_path, '4dof_arm_structure.yaml'),
                'arm_safety_path': os.path.join(arm_data_path, '4dof_arm_safety.yaml'),
                'denorm_config_path': os.path.join(arm_data_path, '4dof_arm_denorm.yaml'),
            }]
    )

    return LaunchDescription([
        arm_control_node
    ])
