from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='/dev/input/js0',
        description='Device path for joystick input'
    )
    
    # Define all nodes
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': LaunchConfiguration('joy_dev'),
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }]
    )
    
    joy_tester_node = Node(
        package='hardware_integration',
        executable='joy_tester',
        name='joy_tester_node'
    )
    
    # Create and return launch description
    return LaunchDescription([
        joy_dev_arg,
        joy_node,
        joy_tester_node
    ])