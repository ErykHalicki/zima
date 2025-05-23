from setuptools import find_packages, setup

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
            'ikpy_test = zima_controls.ikpy_test:main',
            'ik_service = zima_controls.ik_service:main',
            'ik_client_speed_test = zima_controls.ik_client_speed_test:main',
            'arm_controller = zima_controls.arm_controller:main'
        ],
    },
)
