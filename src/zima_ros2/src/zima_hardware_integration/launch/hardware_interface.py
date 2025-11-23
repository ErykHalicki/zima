from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port for communicating with the hardware'
    )
    
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate',
        default_value='115200',
        description='Baud rate for serial communication'
    )
    
    update_rate_arg = DeclareLaunchArgument(
        'hardware_update_rate',
        default_value='200.0',
        description='Rate (Hz) at which to read from serial port'
    )

    output_width_arg = DeclareLaunchArgument(
        'output_width',
        default_value='640',
        description='Camera output width (both width and height required if set)'
    )

    output_height_arg = DeclareLaunchArgument(
        'output_height',
        default_value='480',
        description='Camera output height (both width and height required if set)'
    )

    # Create node configurations
    serial_receiver_node = Node(
        package='zima_hardware_integration',
        executable='serial_receiver',
        name='serial_receiver_node',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'update_rate': LaunchConfiguration('hardware_update_rate'),
            'timeout': 0.1
        }],
        output='screen'
    )
    
    serial_sender_node = Node(
        package='zima_hardware_integration',
        executable='serial_sender',
        name='serial_sender_node',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'timeout': 0.1
        }],
        output='screen'
    )
    
    command_generator_node = Node(
        package='zima_hardware_integration',
        executable='command_generator',
        name='command_generator_node',
        output='screen'
    )

    camera_driver_node = Node(
        package='zima_hardware_integration',
        executable='camera_driver',
        name='camera_publisher',
        parameters=[{
            'output_width': LaunchConfiguration('output_width'),
            'output_height': LaunchConfiguration('output_height')
        }],
        output='screen'
    )

    # Create and return launch description
    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        update_rate_arg,
        output_width_arg,
        output_height_arg,
        serial_receiver_node,
        serial_sender_node,
        command_generator_node,
        camera_driver_node
    ])
