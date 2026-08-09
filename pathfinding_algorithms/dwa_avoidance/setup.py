from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dwa_avoidance'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.lua')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtlebot',
    maintainer_email='jacobr203@gmail.com',
    description='DWA obstacle avoidance with Cartographer SLAM',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa_node = dwa_avoidance.dwa_node:main',
            'scan_relay_node = dwa_avoidance.scan_relay_node:main',
            'odom_relay_node = dwa_avoidance.odom_relay_node:main',
            'tf_relay_node = dwa_avoidance.tf_relay_node:main',
        ],
    },
)
