from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    zima_controls_dir = get_package_share_directory('zima_controls')
    zima_hardware_integration_dir = get_package_share_directory('zima_hardware_integration')
    zima_ml_dir = get_package_share_directory('zima_ml')

    arm_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zima_controls_dir, 'launch', 'arm_control.py')
        )
    )

    hardware_interface_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zima_hardware_integration_dir, 'launch', 'hardware_interface.py')
        )
    )

    teleop_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zima_ml_dir, 'launch', 'teleop_controller.launch.py')
        )
    )

    return LaunchDescription([
        arm_control_launch,
        hardware_interface_launch,
        teleop_controller_launch
    ])
