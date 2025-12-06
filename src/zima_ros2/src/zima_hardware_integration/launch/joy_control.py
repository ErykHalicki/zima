from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    joy_dev_arg = DeclareLaunchArgument(
        'joy_dev',
        default_value='/dev/input/js0',
        description='Device path for joystick input'
    )
    
    joy_config_arg = DeclareLaunchArgument(
        'joy_config',
        default_value='ps3',
        description='Joystick configuration'
    )
    
    joy_deadzone_arg = DeclareLaunchArgument(
        'joy_deadzone',
        default_value='0.05',
        description='Deadzone for joystick inputs (0.0 to 1.0)'
    )
    
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='210',
        description='Maximum speed value (0-255)'
    )
    
    # Define all nodes
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': LaunchConfiguration('joy_dev'),
            'deadzone': LaunchConfiguration('joy_deadzone'),
            'autorepeat_rate': 20.0,
        }]
    )
    
    joy_control_node = Node(
        package='zima_hardware_integration',
        executable='joy_control',
        name='joy_control_node',
        parameters=[{
            'joy_deadzone': LaunchConfiguration('joy_deadzone'),
            'max_speed': LaunchConfiguration('max_speed'),
            'linear_axis': 0,     # Left stick Y for forward/backward
            'angular_axis': 1,    # Left stick X for left/right
            'enable_button': 5,   # R1 button (index 5)
            'emergency_stop_button': 7,  # R2 button (index 7)
        }]
    )
       
    # Create and return launch description
    return LaunchDescription([
        joy_dev_arg,
        joy_config_arg,
        joy_deadzone_arg,
        max_speed_arg,
        joy_node,
        joy_control_node,
    ])
