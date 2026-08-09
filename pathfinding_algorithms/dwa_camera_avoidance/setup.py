from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dwa_camera_avoidance'

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
        (os.path.join('share', package_name, 'models', 'turtlebot3_burger_brio'),
         glob('models/turtlebot3_burger_brio/*')),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtlebot',
    maintainer_email='jacobr203@gmail.com',
    description='DWA + camera-fused obstacle avoidance with Cartographer SLAM',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dwa_camera_node          = dwa_camera_avoidance.dwa_camera_node:main',
            'camera_obstacle_node     = dwa_camera_avoidance.camera_obstacle_node:main',
        ],
    },
)
