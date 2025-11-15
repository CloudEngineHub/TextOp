#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import time


class ResetTester(Node):

    def __init__(self):
        super().__init__('reset_tester')

        # Create publisher for reset command
        self.reset_publisher = self.create_publisher(Empty, '/mjc/reset', 10)

        # Wait for publisher to be ready
        time.sleep(1.0)

        self.get_logger().info(
            'Reset tester node started. Press Ctrl+C to stop.')

        # Create timer to send reset command every 10 seconds
        self.timer = self.create_timer(10.0, self.send_reset_command)

    def send_reset_command(self):
        msg = Empty()
        self.reset_publisher.publish(msg)
        self.get_logger().info('Sent reset command to /mjc/reset')

    def send_manual_reset(self):
        """Send a single reset command manually"""
        msg = Empty()
        self.reset_publisher.publish(msg)
        self.get_logger().info('Manual reset command sent to /mjc/reset')


def main(args=None):
    rclpy.init(args=args)

    node = ResetTester()

    print("=== MuJoCo Reset Tester ===")
    print(
        "This script will send reset commands to /mjc/reset every 10 seconds")
    print("You can also send a manual reset by running:")
    print("  ros2 topic pub --once /mjc/reset std_msgs/msg/Empty")
    print("Press Ctrl+C to stop")
    print()

    try:
        # Send an initial reset command
        node.send_manual_reset()

        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Reset tester stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
