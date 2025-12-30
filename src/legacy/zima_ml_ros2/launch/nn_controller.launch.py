from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    zima_ml_dir = get_package_share_directory('zima_ml')
    default_model_path = os.path.join(zima_ml_dir, 'weights', 'latest.pt')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to the neural network model checkpoint'
    )

    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='210',
        description='Maximum motor speed for neural network controller'
    )

    nn_controller_node = Node(
        package='zima_ml',
        executable='nn_controller',
        name='nn_controller',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'max_speed': LaunchConfiguration('max_speed')
        }]
    )

    return LaunchDescription([
        model_path_arg,
        max_speed_arg,
        nn_controller_node
    ])
