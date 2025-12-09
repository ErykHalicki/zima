from setuptools import find_packages, setup
from glob import glob

package_name = 'zima_controls'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/arm_control.py']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/arm_data', glob('arm_data/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eryk',
    maintainer_email='erykhalicki0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_controller = zima_controls.arm_controller:main'
        ],
    },
)
