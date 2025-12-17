---
sidebar_position: 3
---

# Chapter 1.2: Bridging Digital Agents: Controlling Robots with Python (rclpy)

## Introduction to rclpy

rclpy is the Python client library for ROS 2 (Robot Operating System 2). It provides the Python API for creating ROS 2 nodes, publishers, subscribers, services, and other ROS 2 concepts.

## Core Components of rclpy

### Nodes

In rclpy, a node is created by subclassing `rclpy.node.Node`. This provides all the functionality needed to interact with the ROS 2 ecosystem.

```python
import rclpy
from rclpy.node import Node

class MyRobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        # Node initialization code here
```

### Publishers

Publishers allow nodes to send messages to topics. They are created using the `create_publisher` method:

```python
from std_msgs.msg import String

class MyPublisher(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'robot_commands', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello from robot controller'
        self.publisher.publish(msg)
```

### Subscribers

Subscribers allow nodes to receive messages from topics:

```python
class MySubscriber(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')
```

## Creating a Simple Robot Controller

Here's an example of a basic robot controller that uses both publishing and subscribing:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')

        # Publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.obstacle_detected = False

    def scan_callback(self, msg):
        # Check for obstacles in front of robot
        if len(msg.ranges) > 0:
            front_distance = msg.ranges[len(msg.ranges) // 2]
            self.obstacle_detected = front_distance < 1.0  # 1 meter threshold

    def control_loop(self):
        cmd = Twist()

        if self.obstacle_detected:
            # Stop and turn
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
        else:
            # Move forward
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0

        self.cmd_vel_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

1. **Use appropriate QoS profiles** for different types of data
2. **Handle exceptions** when interacting with ROS 2 APIs
3. **Clean up resources** by properly destroying nodes and other objects
4. **Use parameter declarations** to make nodes configurable
5. **Log appropriately** for debugging and monitoring

## Next Steps

In the next chapter, we'll explore defining robot structure using URDF (Unified Robot Description Format).