from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='traffic_detection',
            executable='traffic_detection',
            name='traffic_detection',
            output='screen',
        )
    ])
