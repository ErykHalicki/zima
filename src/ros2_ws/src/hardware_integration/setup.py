from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hardware_integration'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eryk',
    maintainer_email='erykhalicki0@gmail.com',
    description='ROS2 package for interfacing with Zima robot hardware',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "serial_sender=hardware_integration.serial_sender:main",
            "serial_receiver=hardware_integration.serial_receiver:main",
            "command_generator=hardware_integration.command_generator:main",
        ],
    },
)