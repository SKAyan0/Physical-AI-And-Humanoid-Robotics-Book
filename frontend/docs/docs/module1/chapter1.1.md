---
sidebar_position: 2
---

# Chapter 1.1: Introduction to Middleware: ROS 2 Nodes, Topics, and Services

## What is ROS 2?

ROS 2 (Robot Operating System 2) is flexible framework for writing robot applications. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Core Concepts

### Nodes

A node is an entity that performs computation in the ROS graph. In ROS 2, nodes are implemented as processes that perform computation on behalf of the robot. Multiple nodes can be combined together to form a complete robot application.

### Topics and Publishing/Subscription

Topics are named buses over which nodes exchange messages. The communication is based on a publish/subscribe paradigm where publishers send messages and subscribers receive them.

### Services

Services provide a request/reply communication pattern. A service client sends a request message and waits for a reply message from the service server.

## Code Example

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

This example demonstrates a simple publisher node that sends "Hello World" messages.

## Next Steps

In the next chapter, we'll explore how to bridge digital agents and control robots using Python.