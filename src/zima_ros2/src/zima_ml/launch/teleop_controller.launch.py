from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import os

def generate_launch_description():
    update_rate_arg = DeclareLaunchArgument(
        'teleop_update_rate',
        default_value='30.0',
        description='Rate (Hz) at which to poll teleop control state'
    )

    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='255',
        description='Maximum motor speed for teleop controller'
    )

    dataset_directory_arg = DeclareLaunchArgument(
        'dataset_directory',
        default_value='/home/zima/datasets/',
        description='Directory where datasets are stored'
    )

    dataset_name_arg = DeclareLaunchArgument(
        'dataset_name',
        default_value='default_dataset',
        description='Name of the dataset file'
    )

    dataset_path = PathJoinSubstitution([
        LaunchConfiguration('dataset_directory'),
        LaunchConfiguration('dataset_name')
    ])

    teleop_controller_node = Node(
        package='zima_ml',
        executable='teleop_controller',
        name='teleop_controller',
        parameters=[{
            'update_rate': LaunchConfiguration('teleop_update_rate'),
            'max_speed': LaunchConfiguration('max_speed'),
            'dataset_path': dataset_path
        }],
        output='screen'
    )

    return LaunchDescription([
        update_rate_arg,
        max_speed_arg,
        dataset_directory_arg,
        dataset_name_arg,
        teleop_controller_node
    ])
