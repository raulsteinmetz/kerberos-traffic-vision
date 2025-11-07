from setuptools import setup
import os
from glob import glob

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
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raul',
    maintainer_email='raulsteinmetz0808@gmail.com',
    description='Traffic detection node for kerberos.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'traffic_detection = traffic_detection.traffic_detection:main',
            'test_traffic_detection = traffic_detection.test_traffic_detection:main',
        ],
    },
)
