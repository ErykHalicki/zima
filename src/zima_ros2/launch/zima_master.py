from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directories
    zima_controls_dir = get_package_share_directory('zima_controls')
    zima_hardware_integration_dir = get_package_share_directory('zima_hardware_integration')

    # Include arm control launch file
    arm_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zima_controls_dir, 'launch', 'arm_control.py')
        )
    )

    # Include hardware interface launch file
    hardware_interface_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zima_hardware_integration_dir, 'launch', 'hardware_interface.py')
        )
    )

    return LaunchDescription([
        arm_control_launch,
        hardware_interface_launch
    ])
