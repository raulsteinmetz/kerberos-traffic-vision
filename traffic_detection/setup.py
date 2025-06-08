from setuptools import setup

package_name = 'traffic_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detection.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raul',
    maintainer_email='raulsteinmetz0808@gmail.com',
    description='Minimal ROS 2 node with pub/sub for testing.',
    license='MIT',
    entry_points= {
        'console_scripts': [
            'detection_node = traffic_detection.detection_node:main',
            'test_controller = traffic_detection.test_controller:main',
        ],
    },
)
