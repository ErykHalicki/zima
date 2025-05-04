from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='zima_controls',
            executable='ik_service',
            name='ik_solver',
            output='screen',
            parameters=[
                {'debug_mode': False},
                {'position_error_threshold': 0.07}
            ]
        ),

        Node(
            package='zima_controls',
            executable='arm_controller',
            name='arm_controller',
            output='screen'
        )
    ])
