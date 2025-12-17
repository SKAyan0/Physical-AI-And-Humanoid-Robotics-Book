---
sidebar_position: 4
---

# Chapter 1.3: Defining the Body: URDF for Humanoid Kinematics

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including links, joints, and their relationships.

## URDF Structure

A URDF file consists of several key elements:

- **Links**: Rigid parts of the robot (e.g., base, arms, legs)
- **Joints**: Connections between links that allow relative motion
- **Visual**: How the link appears in visualization tools
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## Basic URDF Example

Here's a simple URDF for a humanoid robot with a base and one arm:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Shoulder link -->
  <link name="shoulder_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting base and shoulder -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Humanoid Robot URDF

A more complete humanoid robot would include multiple joints and links to represent the full body:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
  </joint>

  <!-- Left upper arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left shoulder joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
  </joint>
</robot>
```

## Xacro for Complex URDFs

For complex robots, Xacro (XML Macros) is often used to make URDFs more maintainable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_humanoid">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="arm_length" value="0.3" />
  <xacro:property name="arm_radius" value="0.04" />

  <!-- Macro for arm -->
  <xacro:macro name="arm" params="side parent xyz rpy">
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="${arm_radius}" length="${arm_length}"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 1 1"/>
        </material>
      </visual>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="20" velocity="2"/>
    </joint>
  </xacro:macro>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
  </link>

  <!-- Use macro to create arms -->
  <xacro:arm side="left" parent="torso" xyz="0.15 0 0.2" rpy="0 0 0"/>
  <xacro:arm side="right" parent="torso" xyz="-0.15 0 0.2" rpy="0 0 0"/>
</robot>
```

## Best Practices

1. **Use consistent naming conventions** for links and joints
2. **Define appropriate inertial properties** for physics simulation
3. **Use Xacro macros** to reduce repetition in complex robots
4. **Test URDF files** with visualization tools like RViz
5. **Include safety limits** for joint positions, velocities, and efforts

## Integration with ROS 2

URDF files are typically loaded using robot_state_publisher to broadcast transforms:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('my_robot_description'),
        'urdf',
        'humanoid_robot.urdf'
    )

    # Launch robot_state_publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_file).read()}]
    )

    return LaunchDescription([robot_state_publisher])
```

## Next Steps

In the next module, we'll explore simulation environments and how to bring our URDF models to life in physics simulators.