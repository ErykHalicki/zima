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

    wrist_camera_node = Node(
        package='zima_hardware_integration',
        executable='camera_driver',
        name='wrist_camera_publisher',
        parameters=[{
            'camera_device': 0,
            'output_width': LaunchConfiguration('output_width'),
            'output_height': LaunchConfiguration('output_height')
        }],
        remappings=[
            ('camera/image_raw/compressed', 'wrist_cam/image_raw/compressed')
        ],
        output='screen'
    )

    front_camera_node = Node(
        package='zima_hardware_integration',
        executable='camera_driver',
        name='front_camera_publisher',
        parameters=[{
            'camera_device': 1,
            'output_width': LaunchConfiguration('output_width'),
            'output_height': LaunchConfiguration('output_height')
        }],
        remappings=[
            ('camera/image_raw/compressed', 'front_cam/image_raw/compressed')
        ],
        output='screen'
    )

    tcp_interface_node = Node(
        package='zima_hardware_integration',
        executable='tcp_interface',
        name='tcp_interface_node',
        parameters=[{
            'tcp_port': 5000,
            'tcp_host': '0.0.0.0',
            'max_speed': 210
        }],
        output='screen'
    )

    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        update_rate_arg,
        output_width_arg,
        output_height_arg,
        serial_receiver_node,
        serial_sender_node,
        command_generator_node,
        wrist_camera_node,
        front_camera_node,
        tcp_interface_node
    ])
