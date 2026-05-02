from setuptools import find_packages, setup

package_name = 'ai_robotic_arm_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AI Robotics Dev',
    maintainer_email='user@example.com',
    description='ROS 2 integration for AI Vision Robotic Arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_node = ai_robotic_arm_ros.arm_node:main'
        ],
    },
)
