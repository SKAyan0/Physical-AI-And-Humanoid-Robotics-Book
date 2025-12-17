---
sidebar_position: 4
---

# Chapter 3.3: Movement Logic: Path Planning for Bipeds using Nav2

## Introduction to Bipedal Navigation

Navigation for bipedal robots presents unique challenges compared to wheeled robots. Humanoid robots must maintain balance, handle complex terrain, and plan movements that account for their dynamic nature. This chapter explores specialized navigation techniques for bipedal robots using the ROS 2 Navigation Stack (Nav2) with modifications for bipedal requirements.

## Understanding Bipedal Kinematics

### Center of Mass and Stability

Bipedal robots must maintain their center of mass (CoM) within their support polygon to remain stable. This requires careful path planning and gait generation:

```python
import numpy as np
import math
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class BipedKinematics:
    def __init__(self, robot_height=1.0, leg_length=0.5):
        self.robot_height = robot_height
        self.leg_length = leg_length
        self.support_polygon = []
        self.com_position = np.array([0.0, 0.0, robot_height/2])  # Approximate CoM

    def calculate_support_polygon(self, left_foot, right_foot, foot_width=0.15, foot_length=0.25):
        """Calculate the support polygon based on foot positions"""
        # Calculate vertices of the support polygon (convex hull of both feet)
        vertices = []

        # Left foot vertices
        vertices.append([left_foot.x - foot_width/2, left_foot.y - foot_length/2])
        vertices.append([left_foot.x + foot_width/2, left_foot.y - foot_length/2])
        vertices.append([left_foot.x + foot_width/2, left_foot.y + foot_length/2])
        vertices.append([left_foot.x - foot_width/2, left_foot.y + foot_length/2])

        # Right foot vertices
        vertices.append([right_foot.x - foot_width/2, right_foot.y - foot_length/2])
        vertices.append([right_foot.x + foot_width/2, right_foot.y - foot_length/2])
        vertices.append([right_foot.x + foot_width/2, right_foot.y + foot_length/2])
        vertices.append([right_foot.x - foot_width/2, right_foot.y + foot_length/2])

        # Find convex hull (simplified approach)
        self.support_polygon = self.convex_hull(vertices)
        return self.support_polygon

    def convex_hull(self, points):
        """Simple convex hull calculation (Graham scan)"""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        points = sorted(set(points))
        if len(points) <= 1:
            return points

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Remove last point of each half because it's repeated
        return lower[:-1] + upper[:-1]

    def is_stable(self, com_x, com_y):
        """Check if center of mass is within support polygon"""
        # Use ray casting algorithm to check if point is inside polygon
        x, y = com_x, com_y
        n = len(self.support_polygon)
        inside = False

        p1x, p1y = self.support_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.support_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

class BipedGaitGenerator:
    def __init__(self, step_length=0.3, step_height=0.1, step_duration=0.8):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.current_phase = 0.0  # 0.0 to 1.0
        self.swing_leg = "left"  # Which leg is currently swinging

    def generate_step_trajectory(self, start_pos, target_pos, phase):
        """Generate a smooth trajectory for a step"""
        # Phase should be between 0 and 1
        phase = max(0.0, min(1.0, phase))

        # Linear interpolation for x, y
        x = start_pos[0] + (target_pos[0] - start_pos[0]) * phase
        y = start_pos[1] + (target_pos[1] - start_pos[1]) * phase

        # Parabolic trajectory for z (step height)
        z = start_pos[2]
        if phase < 0.5:
            # Rising phase
            z = start_pos[2] + self.step_height * (1 - math.cos(math.pi * phase * 2)) / 2
        else:
            # Falling phase
            z = start_pos[2] + self.step_height * (1 - math.cos(math.pi * (1 - phase) * 2)) / 2

        return [x, y, z]

    def calculate_ankle_trajectory(self, start_pos, target_pos, phase):
        """Calculate smooth ankle trajectory for natural walking"""
        # Add ankle rotation for more natural movement
        ankle_roll = 0.0  # Side-to-side roll
        ankle_pitch = 0.0  # Forward/backward pitch

        if phase < 0.2:  # Initial lift
            ankle_pitch = -0.1 * (phase / 0.2)
        elif phase > 0.8:  # Final placement
            ankle_pitch = 0.1 * ((phase - 0.8) / 0.2)

        foot_pos = self.generate_step_trajectory(start_pos, target_pos, phase)
        return foot_pos, ankle_roll, ankle_pitch
```

## Nav2 Configuration for Bipedal Robots

### Custom Costmap Layers

Bipedal robots need specialized costmap layers that consider terrain traversability:

```yaml
# Biped-specific costmap configuration
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 10.0
  publish_frequency: 10.0
  static_map: false
  rolling_window: true
  width: 6.0
  height: 6.0
  resolution: 0.05
  plugins:
    - {name: obstacles_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
    - {name: slope_layer, type: "biped_nav_layers::SlopeLayer"}
    - {name: step_height_layer, type: "biped_nav_layers::StepHeightLayer"}

global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 1.0
  static_map: true
  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
    - {name: terrain_traversability_layer, type: "biped_nav_layers::TerrainTraversabilityLayer"}
```

### Custom Costmap Layer Implementation

```cpp
// C++ implementation for custom costmap layers
#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace biped_nav_layers
{

class SlopeLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SlopeLayer() = default;
  virtual void onInitialize();
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y);
  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);

protected:
  void setupDynamicReconfigure(ros::NodeHandle& nh);
  double max_slope_;
  double slope_cost_factor_;
  bool enabled_;
};

void SlopeLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  default_value_ = nav2_costmap_2d::FREE_SPACE;

  // Get parameters
  nh.param("max_slope", max_slope_, 0.3);  // 30% slope
  nh.param("slope_cost_factor", slope_cost_factor_, 3.0);
  nh.param("enabled", enabled_, true);

  // Setup dynamic reconfigure
  setupDynamicReconfigure(nh);
}

void SlopeLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) return;

  // Get elevation data from a height map
  // This is a simplified example - in practice, you'd use actual elevation data
  double min_elevation = 0.0;
  double max_elevation = 0.0;

  // Calculate slope bounds based on elevation changes
  *min_x = std::min(*min_x, robot_x - 1.0);
  *min_y = std::min(*min_y, robot_y - 1.0);
  *max_x = std::max(*max_x, robot_x + 1.0);
  *max_y = std::max(*max_y, robot_y + 1.0);
}

void SlopeLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_ || !current_) return;

  // Calculate slope costs for each cell
  for (int j = min_j; j < max_j; j++) {
    for (int i = min_i; i < max_i; i++) {
      int index = getIndex(i, j);

      // Calculate slope at this cell (simplified)
      double slope = calculateSlope(i, j);

      if (slope > max_slope_) {
        // Mark as lethal obstacle if slope is too steep
        master_grid.setCost(i, j, nav2_costmap_2d::LETHAL_OBSTACLE);
      } else if (slope > max_slope_ * 0.7) {
        // High cost for steep slopes
        unsigned char cost = static_cast<unsigned char>(
          nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE +
          (slope / max_slope_) * slope_cost_factor_ * 50
        );
        master_grid.setCost(i, j, std::min(cost, nav2_costmap_2d::LETHAL_OBSTACLE - 1));
      }
    }
  }
}

double SlopeLayer::calculateSlope(int i, int j)
{
  // Calculate slope based on height differences
  // This would use actual elevation data in practice
  return 0.0; // Simplified implementation
}

PLUGINLIB_EXPORT_CLASS(biped_nav_layers::SlopeLayer, nav2_costmap_2d::Layer)

} // namespace biped_nav_layers
```

## Custom Path Planner for Bipedal Robots

### Biped-Aware Global Planner

```python
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionServer
from rclpy.node import Node
from std_msgs.msg import Header

class BipedGlobalPlanner(Node):
    def __init__(self):
        super().__init__('biped_global_planner')

        # Action server for path computation
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_path_planning
        )

        # Biped-specific parameters
        self.max_step_height = 0.15  # meters
        self.max_slope = 0.3        # 30% grade
        self.footprint_radius = 0.3 # meters
        self.min_turn_radius = 0.5  # meters

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)

        # Costmap subscribers (simplified)
        self.costmap = None
        self.costmap_resolution = 0.05
        self.costmap_origin = [0, 0]

    def execute_path_planning(self, goal_handle):
        """Execute the path planning action"""
        self.get_logger().info('Executing path planning for bipedal robot')

        goal = goal_handle.request.goal
        start = goal_handle.request.start

        # Plan path with biped constraints
        path = self.plan_biped_path(start.pose, goal.pose)

        if path:
            result = ComputePathToPose.Result()
            result.path = path
            goal_handle.succeed()
            self.get_logger().info('Path planning succeeded')
            return result
        else:
            goal_handle.abort()
            self.get_logger().error('Path planning failed')
            return None

    def plan_biped_path(self, start_pose, goal_pose):
        """Plan a path considering bipedal robot constraints"""
        # Convert poses to grid coordinates
        start_grid = self.pose_to_grid(start_pose)
        goal_grid = self.pose_to_grid(goal_pose)

        if not self.is_valid_start_goal(start_grid, goal_grid):
            return None

        # Use modified A* algorithm that considers biped constraints
        path_grid = self.biped_astar(start_grid, goal_grid)

        if path_grid:
            # Convert grid path to pose path
            path_pose = self.grid_path_to_pose_path(path_grid)

            # Smooth path for bipedal movement
            smoothed_path = self.smooth_biped_path(path_pose)

            # Publish the path
            self.path_pub.publish(smoothed_path)

            return smoothed_path

        return None

    def biped_astar(self, start, goal):
        """A* algorithm modified for bipedal robots"""
        import heapq

        # Priority queue: (cost, position)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.biped_heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            current_cost, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Check 8-connected neighbors
            neighbors = self.get_biped_valid_neighbors(current)

            for neighbor in neighbors:
                if neighbor in came_from:  # Already visited
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + self.biped_move_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.biped_heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

    def get_biped_valid_neighbors(self, pos):
        """Get valid neighbors considering bipedal constraints"""
        x, y = pos
        neighbors = []

        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < 1000 and 0 <= ny < 1000):  # Assuming 1000x1000 grid
                    continue

                # Check if cell is traversable for biped
                if self.is_biped_traversable(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def is_biped_traversable(self, x, y):
        """Check if cell is traversable considering bipedal constraints"""
        # Check basic occupancy
        if self.get_cost(x, y) >= 50:  # Occupied
            return False

        # Check slope constraints (simplified)
        # In a real implementation, you'd check the actual terrain slope
        if self.get_slope(x, y) > self.max_slope:
            return False

        # Check step height constraints
        if self.get_step_height(x, y) > self.max_step_height:
            return False

        return True

    def biped_move_cost(self, from_pos, to_pos):
        """Calculate movement cost considering bipedal factors"""
        # Base cost (Euclidean distance)
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        base_cost = math.sqrt(dx*dx + dy*dy)

        # Additional cost for diagonal moves
        if dx != 0 and dy != 0:
            base_cost *= 1.414  # sqrt(2)

        # Additional cost based on terrain difficulty
        terrain_cost = self.get_terrain_cost(to_pos[0], to_pos[1])

        return base_cost + terrain_cost

    def biped_heuristic(self, pos, goal):
        """Heuristic function considering bipedal constraints"""
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        return math.sqrt(dx*dx + dy*dy)  # Simple Euclidean distance

    def get_cost(self, x, y):
        """Get cost value from costmap"""
        # Simplified - in practice, you'd access the actual costmap
        return 0

    def get_slope(self, x, y):
        """Get slope value at position"""
        # Simplified - in practice, you'd access elevation data
        return 0.0

    def get_step_height(self, x, y):
        """Get step height at position"""
        # Simplified - in practice, you'd access elevation data
        return 0.0

    def get_terrain_cost(self, x, y):
        """Get additional terrain cost"""
        # Simplified - in practice, you'd consider surface type, roughness, etc.
        return 0.0

    def pose_to_grid(self, pose):
        """Convert pose to grid coordinates"""
        x_grid = int((pose.position.x - self.costmap_origin[0]) / self.costmap_resolution)
        y_grid = int((pose.position.y - self.costmap_origin[1]) / self.costmap_resolution)
        return (x_grid, y_grid)

    def grid_path_to_pose_path(self, grid_path):
        """Convert grid path to PoseStamped path"""
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        for grid_x, grid_y in grid_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = grid_x * self.costmap_resolution + self.costmap_origin[0]
            pose_stamped.pose.position.y = grid_y * self.costmap_resolution + self.costmap_origin[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0  # No rotation

            path.poses.append(pose_stamped)

        return path

    def smooth_biped_path(self, path):
        """Smooth path for bipedal robot movement"""
        if len(path.poses) < 3:
            return path

        # Implement path smoothing that considers bipedal dynamics
        smoothed_poses = [path.poses[0]]  # Start with first pose

        i = 0
        while i < len(path.poses) - 1:
            # Try to find the furthest point that can be reached directly
            j = len(path.poses) - 1

            while j > i + 1:
                if self.is_direct_path_biped_traversable(path.poses[i], path.poses[j]):
                    smoothed_poses.append(path.poses[j])
                    i = j
                    break
                j -= 1

            if j == i + 1:  # No intermediate point found, add next point
                smoothed_poses.append(path.poses[i + 1])
                i += 1

        # Create new path with smoothed poses
        smoothed_path = Path()
        smoothed_path.header = path.header
        smoothed_path.poses = smoothed_poses

        return smoothed_path

    def is_direct_path_biped_traversable(self, start, end):
        """Check if direct path between two poses is traversable for biped"""
        # Use line-of-sight check with biped constraints
        start_point = (start.pose.position.x, start.pose.position.y)
        end_point = (end.pose.position.x, end.pose.position.y)

        # Check intermediate points along the line
        steps = max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])) / self.costmap_resolution
        steps = max(int(steps), 10)  # At least 10 checks

        for i in range(steps + 1):
            t = i / steps
            x = start_point[0] + (end_point[0] - start_point[0]) * t
            y = start_point[1] + (end_point[1] - start_point[1]) * t

            grid_x = int((x - self.costmap_origin[0]) / self.costmap_resolution)
            grid_y = int((y - self.costmap_origin[1]) / self.costmap_resolution)

            if not self.is_biped_traversable(grid_x, grid_y):
                return False

        return True

    def is_valid_start_goal(self, start, goal):
        """Check if start and goal positions are valid"""
        return (self.is_biped_traversable(start[0], start[1]) and
                self.is_biped_traversable(goal[0], goal[1]))
```

## Bipedal Local Planner

### Footstep Planner

```python
import numpy as np
from geometry_msgs.msg import Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, max_step_yaw=0.3):
        self.step_length = step_length
        self.step_width = step_width
        self.max_step_yaw = max_step_yaw  # Maximum yaw change per step (radians)

        # Robot configuration
        self.left_foot_pose = Pose()
        self.right_foot_pose = Pose()
        self.support_foot = "left"  # Which foot is currently supporting weight

    def plan_footsteps(self, path_poses, start_left_foot, start_right_foot):
        """Plan footstep sequence to follow the given path"""
        footsteps = []

        # Initialize foot positions
        self.left_foot_pose = start_left_foot
        self.right_foot_pose = start_right_foot

        # Follow the path with alternating footsteps
        for i, target_pose in enumerate(path_poses):
            # Calculate next foot position based on path direction
            next_left, next_right = self.calculate_next_foot_positions(
                target_pose,
                self.left_foot_pose,
                self.right_foot_pose
            )

            # Determine which foot to move
            if self.support_foot == "left":
                # Move right foot
                if self.is_step_valid(self.right_foot_pose, next_right):
                    self.right_foot_pose = next_right
                    footsteps.append(("right", next_right))
                    self.support_foot = "right"
            else:
                # Move left foot
                if self.is_step_valid(self.left_foot_pose, next_left):
                    self.left_foot_pose = next_left
                    footsteps.append(("left", next_left))
                    self.support_foot = "left"

        return footsteps

    def calculate_next_foot_positions(self, target_pose, left_foot, right_foot):
        """Calculate next positions for both feet based on target"""
        # Calculate desired position and orientation
        desired_x = target_pose.position.x
        desired_y = target_pose.position.y

        # For simplicity, assume we want to step in the direction of the path
        # In practice, you'd use more sophisticated gait planning

        # Calculate step direction (simplified)
        if self.support_foot == "left":
            # Right foot should move toward target
            dx = desired_x - right_foot.position.x
            dy = desired_y - right_foot.position.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.step_length * 0.5:  # Only step if far enough
                # Normalize and scale to step length
                scale = min(self.step_length, dist) / dist
                new_x = right_foot.position.x + dx * scale
                new_y = right_foot.position.y + dy * scale

                new_right = Pose()
                new_right.position.x = new_x
                new_right.position.y = new_y
                new_right.position.z = right_foot.position.z  # Maintain height
                new_right.orientation = target_pose.orientation

                return left_foot, new_right
        else:
            # Left foot should move toward target
            dx = desired_x - left_foot.position.x
            dy = desired_y - left_foot.position.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.step_length * 0.5:  # Only step if far enough
                # Normalize and scale to step length
                scale = min(self.step_length, dist) / dist
                new_x = left_foot.position.x + dx * scale
                new_y = left_foot.position.y + dy * scale

                new_left = Pose()
                new_left.position.x = new_x
                new_left.position.y = new_y
                new_left.position.z = left_foot.position.z  # Maintain height
                new_left.orientation = target_pose.orientation

                return new_left, right_foot

        return left_foot, right_foot  # No change if not far enough

    def is_step_valid(self, current_pose, next_pose):
        """Check if a step is kinematically and dynamically valid"""
        # Check distance (should not be too far)
        dx = next_pose.position.x - current_pose.position.x
        dy = next_pose.position.y - current_pose.position.y
        step_distance = math.sqrt(dx*dx + dy*dy)

        if step_distance > self.step_length * 1.5:  # Allow some flexibility
            return False

        # Check yaw change
        # For simplicity, we'll assume yaw is valid
        # In practice, you'd check actual orientation constraints

        # Check height change (step height)
        dz = abs(next_pose.position.z - current_pose.position.z)
        if dz > 0.2:  # Maximum step height
            return False

        return True

class BipedController:
    def __init__(self):
        # Initialize biped-specific controllers
        self.balance_controller = BalanceController()
        self.footstep_planner = FootstepPlanner()
        self.path_follower = BipedPathFollower()

    def execute_navigation(self, path, initial_left_foot, initial_right_foot):
        """Execute navigation for bipedal robot"""
        # Plan footstep sequence
        footsteps = self.footstep_planner.plan_footsteps(
            path.poses,
            initial_left_foot,
            initial_right_foot
        )

        # Execute footstep plan with balance control
        for foot, pose in footsteps:
            # Move foot to new position with balance maintenance
            success = self.execute_footstep(foot, pose)

            if not success:
                # Handle failure - maybe replan
                return False

            # Update balance after each step
            self.balance_controller.update_support_polygon(foot, pose)

        return True

    def execute_footstep(self, foot, target_pose):
        """Execute a single footstep with balance control"""
        # This would interface with the robot's joint controllers
        # and balance maintenance systems

        # For simulation purposes:
        self.get_logger().info(f'Moving {foot} foot to {target_pose.position.x}, {target_pose.position.y}')

        # In a real system, you'd:
        # 1. Generate joint trajectories for the step
        # 2. Maintain balance by adjusting other joints
        # 3. Execute the motion
        # 4. Verify success

        return True  # Simplified success
```

## Integration with Nav2 Behavior Trees

### Custom Behavior Tree Nodes

```xml
<!-- Custom behavior tree for bipedal navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence name="NavigateWithBiped">
            <GoalReached goal_checker="biped_goal_checker"/>
            <ComputePathToPoseBiped path_publisher="global_costmap/global_costmap/footprint"/>
            <SmoothPathBiped/>
            <FollowPathBiped local_planner="biped_local_planner"/>
        </Sequence>
    </BehaviorTree>

    <BehaviorTree ID="RecoveryBiped">
        <ReactiveSequence name="RecoveryNodeBiped">
            <ClearEntirelyCostmapBiped name="ClearLocalCostmapBiped" service_name="local_costmap/clear_entirely_local_costmap"/>
            <ClearEntirelyCostmapBiped name="ClearGlobalCostmapBiped" service_name="global_costmap/clear_entirely_global_costmap"/>
            <SpinBiped spin_dist="1.57"/>
            <WaitBiped wait_duration="5"/>
        </ReactiveSequence>
    </BehaviorTree>
</root>
```

## Practical Implementation Example

### Complete Biped Navigation Node

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionServer, ActionClient
from rclpy.qos import QoSProfile
import numpy as np
import math

class BipedNavigationNode(Node):
    def __init__(self):
        super().__init__('biped_navigation')

        # Action server for navigation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_navigation
        )

        # Publishers and subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )

        self.path_pub = self.create_publisher(Path, 'biped_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Navigation state
        self.current_pose = None
        self.current_twist = None
        self.current_imu = None

        # Biped-specific parameters
        self.step_length = 0.3
        self.max_linear_speed = 0.3  # m/s
        self.max_angular_speed = 0.5  # rad/s
        self.min_step_time = 0.5  # seconds between steps

        # Path following parameters
        self.lookahead_distance = 0.5
        self.arrival_threshold = 0.2

        # Initialize controllers
        self.path_planner = BipedGlobalPlanner()
        self.footstep_planner = FootstepPlanner()
        self.controller = BipedController()

        # Foot pose estimation (simplified)
        self.left_foot_pose = Pose()
        self.right_foot_pose = Pose()

        self.get_logger().info('Biped Navigation Node initialized')

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Update foot pose estimates based on odometry
        # This is a simplified approach
        self.update_foot_poses_from_odom(msg)

    def imu_callback(self, msg):
        """Handle IMU updates for balance control"""
        self.current_imu = msg

        # Use IMU data for balance feedback
        self.balance_feedback(msg)

    def update_foot_poses_from_odom(self, odom_msg):
        """Update estimated foot poses based on odometry"""
        # Simplified estimation - in practice, you'd use forward kinematics
        # or actual sensor data

        # For now, just use the robot's position as a reference
        robot_x = odom_msg.pose.pose.position.x
        robot_y = odom_msg.pose.pose.position.y

        # Estimate foot positions relative to robot
        self.left_foot_pose.position.x = robot_x + 0.1  # Offset to the left
        self.left_foot_pose.position.y = robot_y + 0.1
        self.left_foot_pose.position.z = 0.0  # On the ground

        self.right_foot_pose.position.x = robot_x - 0.1  # Offset to the right
        self.right_foot_pose.position.y = robot_y + 0.1
        self.right_foot_pose.position.z = 0.0

    def balance_feedback(self, imu_msg):
        """Use IMU data for balance feedback"""
        # Extract orientation from IMU
        orientation = imu_msg.orientation
        angular_velocity = imu_msg.angular_velocity

        # Implement balance control logic
        # This would interface with the robot's balance controller
        self.adjust_balance(orientation, angular_velocity)

    def adjust_balance(self, orientation, angular_velocity):
        """Adjust robot's balance based on IMU feedback"""
        # Simplified balance adjustment
        # In practice, you'd implement PID controllers or other balance algorithms

        # Example: adjust based on roll and pitch angles
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # If the robot is tilting too much, adjust stance
        max_tilt = 0.1  # 5.7 degrees

        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            self.get_logger().warn(f'Balance adjustment needed: roll={roll}, pitch={pitch}')
            # Implement balance recovery logic here

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def execute_navigation(self, goal_handle):
        """Execute the navigation action"""
        self.get_logger().info('Executing bipedal navigation')

        goal = goal_handle.request.pose
        start_pose = PoseStamped()
        start_pose.pose = self.current_pose if self.current_pose else goal.pose

        # Plan path with biped constraints
        path = self.path_planner.plan_biped_path(start_pose.pose, goal.pose)

        if not path:
            self.get_logger().error('Failed to plan path')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Execute the planned path
        success = self.follow_path(path)

        if success:
            goal_handle.succeed()
            result = NavigateToPose.Result()
            self.get_logger().info('Navigation succeeded')
            return result
        else:
            goal_handle.abort()
            self.get_logger().error('Navigation failed')
            return NavigateToPose.Result()

    def follow_path(self, path):
        """Follow the planned path with biped-specific control"""
        if not path.poses:
            return True

        # Plan footstep sequence
        footsteps = self.footstep_planner.plan_footsteps(
            path.poses,
            self.left_foot_pose,
            self.right_foot_pose
        )

        if not footsteps:
            return False

        # Execute footsteps
        for foot, pose in footsteps:
            success = self.execute_footstep(foot, pose)
            if not success:
                return False

            # Small delay between steps
            time.sleep(self.min_step_time)

        return True

    def execute_footstep(self, foot, target_pose):
        """Execute a single footstep"""
        self.get_logger().info(f'Executing {foot} footstep to ({target_pose.position.x}, {target_pose.position.y})')

        # In a real implementation, this would:
        # 1. Generate smooth trajectories for the foot
        # 2. Maintain balance by adjusting other joints
        # 3. Execute the motion using joint controllers
        # 4. Verify successful completion

        # For simulation, just return success
        return True

def main(args=None):
    rclpy.init(args=args)

    try:
        biped_nav_node = BipedNavigationNode()
        rclpy.spin(biped_nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        biped_nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Bipedal Navigation

1. **Stability First**: Always prioritize robot stability over path optimality
2. **Gradual Changes**: Make gradual adjustments to avoid dynamic instabilities
3. **Sensor Fusion**: Combine multiple sensors (IMU, encoders, cameras) for robust state estimation
4. **Adaptive Parameters**: Adjust navigation parameters based on terrain and robot state
5. **Fallback Strategies**: Implement safe fallback behaviors when navigation fails

## Performance Optimization

For real-time bipedal navigation, consider these optimization techniques:

1. **Multi-rate Control**: Run balance control at higher frequency than path planning
2. **Predictive Control**: Use model predictive control for better dynamic response
3. **Terrain Classification**: Pre-classify terrain types to adjust parameters accordingly
4. **Efficient Path Smoothing**: Use computationally efficient path smoothing algorithms

## Next Steps

In the next module, we'll explore Vision-Language-Action (VLA) models and how they integrate with bipedal robots for cognitive planning and natural language interaction.