from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    update_rate_arg = DeclareLaunchArgument(
        'update_rate',
        default_value='50.0',
        description='Rate (Hz) at which to poll teleop control state'
    )

    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='255',
        description='Maximum motor speed for teleop controller'
    )

    teleop_controller_node = Node(
        package='zima_ml',
        executable='teleop_controller',
        name='teleop_controller',
        parameters=[{
            'update_rate': LaunchConfiguration('update_rate'),
            'max_speed': LaunchConfiguration('max_speed')
        }],
        output='screen'
    )

    return LaunchDescription([
        update_rate_arg,
        max_speed_arg,
        teleop_controller_node
    ])
