---
sidebar_position: 3
---

# Chapter 4.2: Cognitive Planning: LLMs as Action Planners (Natural Language to ROS 2)

## Introduction to Cognitive Planning with LLMs

Cognitive planning for robots involves the ability to understand high-level goals expressed in natural language and translate them into executable action sequences. Large Language Models (LLMs) have shown remarkable capabilities in understanding and generating human language, making them ideal candidates for bridging the gap between natural language commands and robotic actions.

## Architecture for LLM-Based Action Planning

### System Overview

The cognitive planning system consists of several components that work together to interpret natural language and generate robot actions:

```python
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import json
import re
from typing import Dict, List, Tuple, Optional
import asyncio
import logging

class LLMActionPlanner(Node):
    """
    LLM-based cognitive planner that translates natural language to ROS 2 actions
    """
    def __init__(self):
        super().__init__('llm_action_planner')

        # OpenAI API configuration
        self.openai_client = None
        self.api_key = self.declare_parameter('openai_api_key', '').value
        if self.api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key)

        # ROS 2 interfaces
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10
        )

        self.response_pub = self.create_publisher(
            String, 'cognitive_planning_response', 10
        )

        # Action clients for robot capabilities
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, FollowJointTrajectory, 'manipulation_controller/follow_joint_trajectory')

        # Robot capabilities database
        self.robot_capabilities = self.initialize_capabilities()

        # Context management
        self.conversation_history = []
        self.max_history_length = 10

        # Logging
        self.logger = self.get_logger()

    def initialize_capabilities(self) -> Dict:
        """
        Initialize the robot's capabilities database
        """
        return {
            "navigation": {
                "description": "Move the robot to specified locations",
                "parameters": {
                    "target_location": {
                        "type": "string",
                        "description": "Target location (e.g., 'kitchen', 'office', 'charging station')"
                    },
                    "target_pose": {
                        "type": "Pose",
                        "description": "Target pose in map coordinates"
                    }
                }
            },
            "manipulation": {
                "description": "Manipulate objects using robot arms",
                "parameters": {
                    "action": {
                        "type": "string",
                        "enum": ["pick", "place", "grasp", "release", "move"],
                        "description": "Manipulation action to perform"
                    },
                    "object": {
                        "type": "string",
                        "description": "Object to manipulate"
                    },
                    "target_location": {
                        "type": "string",
                        "description": "Target location for manipulation"
                    }
                }
            },
            "perception": {
                "description": "Perceive and identify objects in the environment",
                "parameters": {
                    "target_object": {
                        "type": "string",
                        "description": "Object to search for"
                    },
                    "search_area": {
                        "type": "string",
                        "description": "Area to search in"
                    }
                }
            },
            "communication": {
                "description": "Communicate with humans",
                "parameters": {
                    "message": {
                        "type": "string",
                        "description": "Message to communicate"
                    }
                }
            }
        }

    def command_callback(self, msg: String):
        """
        Handle incoming natural language commands
        """
        command = msg.data
        self.logger.info(f"Received command: {command}")

        # Process the command asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self.process_command_async(command),
            asyncio.new_event_loop()
        )

        # Publish response
        response_msg = String()
        response_msg.data = future.result()
        self.response_pub.publish(response_msg)

    async def process_command_async(self, command: str) -> str:
        """
        Process natural language command using LLM
        """
        try:
            # Generate structured plan using LLM
            plan = await self.generate_action_plan(command)

            if plan:
                # Execute the plan
                execution_result = await self.execute_plan(plan)
                return f"Plan executed successfully: {execution_result}"
            else:
                return "Could not generate a valid action plan"
        except Exception as e:
            self.logger.error(f"Error processing command: {str(e)}")
            return f"Error processing command: {str(e)}"

    async def generate_action_plan(self, command: str) -> Optional[List[Dict]]:
        """
        Generate action plan using LLM
        """
        if not self.openai_client:
            self.logger.error("OpenAI client not initialized")
            return None

        # Create a detailed prompt for the LLM
        prompt = self.create_planning_prompt(command)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are a cognitive planning assistant for a humanoid robot. Your job is to interpret natural language commands and generate executable action plans.

                        Robot capabilities: {json.dumps(self.robot_capabilities, indent=2)}

                        Rules:
                        1. Always return a valid JSON response
                        2. Each action should be one of the available capabilities
                        3. Include all required parameters for each action
                        4. Break down complex commands into simpler steps
                        5. Consider the robot's physical limitations
                        """
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            # Parse the response
            plan_text = response.choices[0].message.content
            plan = json.loads(plan_text)

            # Validate the plan structure
            if 'actions' in plan:
                return plan['actions']
            else:
                self.logger.error(f"Invalid plan format: {plan}")
                return None

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error calling LLM: {str(e)}")
            return None

    def create_planning_prompt(self, command: str) -> str:
        """
        Create a detailed prompt for the LLM
        """
        return f"""
        Command: "{command}"

        Generate an action plan for the robot to execute this command. The plan should be a JSON object with an "actions" array containing individual action objects.

        Each action object should have:
        - "action_type": The type of action (one of: {list(self.robot_capabilities.keys())})
        - "parameters": Required parameters for the action
        - "description": Brief description of what this action does

        Example response format:
        {{
            "actions": [
                {{
                    "action_type": "navigation",
                    "parameters": {{
                        "target_location": "kitchen"
                    }},
                    "description": "Navigate to the kitchen"
                }},
                {{
                    "action_type": "perception",
                    "parameters": {{
                        "target_object": "water bottle"
                    }},
                    "description": "Look for a water bottle"
                }}
            ]
        }}

        Generate the plan now:
        """

    async def execute_plan(self, plan: List[Dict]) -> str:
        """
        Execute the action plan
        """
        results = []

        for i, action in enumerate(plan):
            self.logger.info(f"Executing action {i+1}/{len(plan)}: {action.get('description', 'Unknown action')}")

            try:
                result = await self.execute_single_action(action)
                results.append({
                    "action": action,
                    "result": result,
                    "success": True
                })

                # Small delay between actions to allow for stabilization
                await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Action failed: {str(e)}"
                results.append({
                    "action": action,
                    "result": error_msg,
                    "success": False
                })
                self.logger.error(error_msg)

                # For now, continue with the plan even if one action fails
                # In practice, you might want more sophisticated error handling

        return f"Plan completed with {len([r for r in results if r['success']])}/{len(results)} actions successful"

    async def execute_single_action(self, action: Dict) -> str:
        """
        Execute a single action based on its type
        """
        action_type = action.get('action_type')
        parameters = action.get('parameters', {})

        if action_type == 'navigation':
            return await self.execute_navigation_action(parameters)
        elif action_type == 'manipulation':
            return await self.execute_manipulation_action(parameters)
        elif action_type == 'perception':
            return await self.execute_perception_action(parameters)
        elif action_type == 'communication':
            return await self.execute_communication_action(parameters)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def execute_navigation_action(self, params: Dict) -> str:
        """
        Execute navigation action
        """
        # Convert natural language location to coordinates
        # This would typically use a map or location database
        target_pose = await self.resolve_location_to_pose(params.get('target_location'))

        if not target_pose:
            # If location resolution fails, try to parse coordinates from parameters
            if 'target_pose' in params:
                target_pose = self.dict_to_pose(params['target_pose'])
            else:
                raise ValueError(f"Could not resolve location: {params.get('target_location')}")

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Navigation action server not available")

        # Send goal and wait for result
        future = self.nav_client.send_goal_async(goal_msg)
        goal_handle = await future

        if not goal_handle.accepted:
            raise RuntimeError("Navigation goal was rejected")

        result_future = goal_handle.get_result_async()
        result = await result_future

        if result.result:
            return f"Successfully navigated to location"
        else:
            raise RuntimeError("Navigation failed")

    async def execute_manipulation_action(self, params: Dict) -> str:
        """
        Execute manipulation action
        """
        action = params.get('action')
        obj = params.get('object')
        target_location = params.get('target_location')

        # This is a simplified implementation
        # In practice, you'd need more sophisticated manipulation planning
        if action == 'pick' and obj:
            return f"Attempting to pick up {obj}"
        elif action == 'place' and target_location:
            return f"Attempting to place object at {target_location}"
        elif action == 'grasp':
            return f"Attempting to grasp object"
        elif action == 'release':
            return f"Releasing object"
        else:
            raise ValueError(f"Invalid manipulation action: {action}")

    async def execute_perception_action(self, params: Dict) -> str:
        """
        Execute perception action
        """
        target_object = params.get('target_object')
        search_area = params.get('search_area', 'current area')

        # This would interface with perception systems
        # For now, return a mock response
        return f"Searching for {target_object} in {search_area} - object detected"

    async def execute_communication_action(self, params: Dict) -> str:
        """
        Execute communication action
        """
        message = params.get('message', 'Hello')

        # Publish the message to a speech synthesis system
        # This is a simplified implementation
        speech_pub = self.create_publisher(String, 'tts_input', 10)
        msg = String()
        msg.data = message
        speech_pub.publish(msg)

        return f"Communicated: {message}"

    async def resolve_location_to_pose(self, location_name: str) -> Optional[Pose]:
        """
        Resolve a natural language location to a Pose in the map
        """
        # In practice, this would query a location database or semantic map
        # For this example, we'll use a simple mapping
        location_map = {
            "kitchen": Pose(position=Point(x=5.0, y=2.0, z=0.0)),
            "office": Pose(position=Point(x=1.0, y=8.0, z=0.0)),
            "living room": Pose(position=Point(x=3.0, y=5.0, z=0.0)),
            "bedroom": Pose(position=Point(x=8.0, y=3.0, z=0.0)),
            "charging station": Pose(position=Point(x=0.0, y=0.0, z=0.0)),
        }

        return location_map.get(location_name.lower())

    def dict_to_pose(self, pose_dict: Dict) -> Pose:
        """
        Convert a dictionary to a Pose message
        """
        pose = Pose()
        if 'position' in pose_dict:
            pos = pose_dict['position']
            pose.position.x = pos.get('x', 0.0)
            pose.position.y = pos.get('y', 0.0)
            pose.position.z = pos.get('z', 0.0)

        if 'orientation' in pose_dict:
            orient = pose_dict['orientation']
            pose.orientation.x = orient.get('x', 0.0)
            pose.orientation.y = orient.get('y', 0.0)
            pose.orientation.z = orient.get('z', 0.0)
            pose.orientation.w = orient.get('w', 1.0)

        return pose


class LLMRefinementNode(Node):
    """
    Node that refines and validates LLM-generated plans
    """
    def __init__(self):
        super().__init__('llm_refinement_node')

        # Subscriptions and publications
        self.plan_sub = self.create_subscription(
            String, 'raw_plan', self.plan_callback, 10
        )

        self.refined_plan_pub = self.create_publisher(
            String, 'refined_plan', 10
        )

        # Robot kinematic/dynamic constraints
        self.max_velocity = 0.5  # m/s
        self.max_angular_velocity = 0.5  # rad/s
        self.robot_radius = 0.3  # m

    def plan_callback(self, msg: String):
        """
        Receive and refine LLM-generated plan
        """
        try:
            raw_plan = json.loads(msg.data)
            refined_plan = self.refine_plan(raw_plan)

            refined_msg = String()
            refined_msg.data = json.dumps(refined_plan)
            self.refined_plan_pub.publish(refined_msg)

            self.get_logger().info("Plan refined successfully")
        except Exception as e:
            self.get_logger().error(f"Error refining plan: {str(e)}")

    def refine_plan(self, raw_plan: Dict) -> Dict:
        """
        Refine the plan based on robot constraints and safety considerations
        """
        refined_plan = raw_plan.copy()

        # Validate each action against robot constraints
        for action in refined_plan.get('actions', []):
            self.validate_action(action)

        # Add safety checks and intermediate waypoints if needed
        refined_plan = self.add_safety_constraints(refined_plan)

        # Optimize for efficiency while maintaining safety
        refined_plan = self.optimize_plan(refined_plan)

        return refined_plan

    def validate_action(self, action: Dict):
        """
        Validate action against robot constraints
        """
        action_type = action.get('action_type')
        params = action.get('parameters', {})

        if action_type == 'navigation':
            # Validate navigation parameters
            target_pose = params.get('target_pose')
            if target_pose:
                # Check if target is within operational bounds
                # Check for collisions with known obstacles
                pass

    def add_safety_constraints(self, plan: Dict) -> Dict:
        """
        Add safety constraints to the plan
        """
        # Add collision avoidance waypoints
        # Add emergency stop conditions
        # Add validation steps between actions
        return plan

    def optimize_plan(self, plan: Dict) -> Dict:
        """
        Optimize the plan for efficiency
        """
        # Combine similar actions
        # Optimize path for shortest distance/time
        # Consider robot dynamics for smoother execution
        return plan


def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    planner_node = LLMActionPlanner()
    refinement_node = LLMRefinementNode()

    # Use multi-threaded executor to handle callbacks properly
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(planner_node)
    executor.add_node(refinement_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        refinement_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Prompt Engineering for Better Planning

### Effective Prompt Design

The quality of the LLM's output heavily depends on the prompt design. Here's how to create effective prompts for cognitive planning:

```python
class PromptEngineering:
    """
    Class for creating effective prompts for LLM-based cognitive planning
    """

    @staticmethod
    def create_domain_specific_prompt(robot_type: str, capabilities: Dict, environment: str) -> str:
        """
        Create a domain-specific prompt for a particular robot and environment
        """
        return f"""
        You are an AI assistant for a {robot_type} operating in a {environment} environment.

        ROBOT CAPABILITIES:
        {json.dumps(capabilities, indent=2)}

        ENVIRONMENT CONSTRAINTS:
        - Maximum speed limits
        - Obstacle avoidance required
        - Human safety protocols must be followed
        - Battery/energy considerations

        PLANNING RULES:
        1. Always prioritize safety over task completion
        2. Consider the robot's physical limitations
        3. Break down complex tasks into simple, executable steps
        4. Include error handling and recovery steps when possible
        5. Use available capabilities efficiently

        RESPONSE FORMAT:
        {{
            "actions": [
                {{
                    "action_type": "navigation|manipulation|perception|communication",
                    "parameters": {{}},
                    "description": "Brief description of the action",
                    "confidence": 0.0-1.0,
                    "estimated_time": seconds
                }}
            ],
            "reasoning": "Brief explanation of the planning decision",
            "potential_issues": ["list of potential issues"]
        }}

        Remember: Be precise, safe, and consider the robot's limitations.
        """

    @staticmethod
    def create_context_aware_prompt(command: str, context: Dict) -> str:
        """
        Create a prompt that considers the current context
        """
        context_str = json.dumps(context, indent=2)

        return f"""
        COMMAND: "{command}"

        CURRENT CONTEXT:
        {context_str}

        Based on the current context, generate an appropriate action plan that:
        1. Takes into account the current state
        2. Considers recent interactions
        3. Adapts to the current situation
        4. Maintains consistency with previous commands when appropriate

        ACTION PLAN:
        """

    @staticmethod
    def create_multi_step_prompt(command: str, robot_state: Dict) -> str:
        """
        Create a prompt for complex multi-step planning
        """
        return f"""
        COMPLEX COMMAND: "{command}"

        ROBOT STATE: {json.dumps(robot_state, indent=2)}

        Task: Break down this complex command into a sequence of simple, executable actions.

        For each action, consider:
        1. Pre-conditions: What must be true before executing
        2. Effects: What changes after execution
        3. Resources needed: What capabilities are required
        4. Dependencies: What other actions must complete first

        Provide the plan as a directed acyclic graph of actions with dependencies.

        RESPONSE FORMAT:
        {{
            "actions": [
                {{
                    "id": "unique_id",
                    "action_type": "type",
                    "parameters": {{}},
                    "preconditions": [],
                    "effects": [],
                    "dependencies": ["other_action_ids"]
                }}
            ],
            "execution_order": ["action_id_1", "action_id_2", ...]
        }}
        """


class SafetyValidator:
    """
    Class to validate LLM-generated plans for safety
    """

    def __init__(self):
        self.safety_rules = [
            self._no_collision_rule,
            self._speed_limit_rule,
            self._human_proximity_rule,
            self._energy_constraint_rule
        ]

    def validate_plan(self, plan: Dict, current_state: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a plan against safety rules
        """
        violations = []

        for rule in self.safety_rules:
            rule_violations = rule(plan, current_state)
            violations.extend(rule_violations)

        return len(violations) == 0, violations

    def _no_collision_rule(self, plan: Dict, current_state: Dict) -> List[str]:
        """
        Check for potential collisions
        """
        violations = []

        # Check navigation actions for collision with known obstacles
        for action in plan.get('actions', []):
            if action.get('action_type') == 'navigation':
                target = action.get('parameters', {}).get('target_location')
                # Check if path to target is clear
                # This would interface with the navigation stack
                pass

        return violations

    def _speed_limit_rule(self, plan: Dict, current_state: Dict) -> List[str]:
        """
        Check for speed limit violations
        """
        violations = []

        for action in plan.get('actions', []):
            # Check if action parameters exceed speed limits
            pass

        return violations

    def _human_proximity_rule(self, plan: Dict, current_state: Dict) -> List[str]:
        """
        Check for human safety violations
        """
        violations = []

        # Check if any actions would bring robot too close to humans
        pass

        return violations

    def _energy_constraint_rule(self, plan: Dict, current_state: Dict) -> List[str]:
        """
        Check for energy/battery constraints
        """
        violations = []

        # Check if plan would deplete battery before completion
        pass

        return violations
```

## Integration with ROS 2 Ecosystem

### Action Server Integration

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav2_msgs.action import NavigateToPose
from control_msgs.action import FollowJointTrajectory
import threading

class CognitivePlanningActionServer:
    """
    Action server that integrates LLM-based planning
    """

    def __init__(self, node: LLMActionPlanner):
        self.node = node
        self._action_server = ActionServer(
            node,
            ExecuteCognitivePlan,
            'execute_cognitive_plan',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self._goal_handles = {}
        self._executor = asyncio.new_event_loop()
        self._executor_thread = threading.Thread(target=self._run_executor, daemon=True)
        self._executor_thread.start()

    def _run_executor(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self._executor)
        self._executor.run_forever()

    def goal_callback(self, goal_request):
        """Accept or reject a goal request"""
        self.node.get_logger().info(f"Received cognitive planning goal: {goal_request.command}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a cancel request"""
        self.node.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle: ServerGoalHandle):
        """Execute the goal"""
        self.node.get_logger().info("Executing cognitive planning goal")

        try:
            # Generate plan using LLM
            plan = await self.node.generate_action_plan(goal_handle.request.command)

            if not plan:
                goal_handle.abort()
                result = ExecuteCognitivePlan.Result()
                result.success = False
                result.message = "Could not generate valid plan"
                return result

            # Execute the plan
            execution_result = await self.node.execute_plan(plan)

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = ExecuteCognitivePlan.Result()
                result.success = False
                result.message = "Goal canceled during execution"
                return result

            # Complete successfully
            goal_handle.succeed()
            result = ExecuteCognitivePlan.Result()
            result.success = True
            result.message = execution_result
            result.executed_actions = [str(action) for action in plan]

            return result

        except Exception as e:
            self.node.get_logger().error(f"Error executing cognitive plan: {str(e)}")
            goal_handle.abort()
            result = ExecuteCognitivePlan.Result()
            result.success = False
            result.message = f"Execution failed: {str(e)}"
            return result
```

## Handling Uncertainty and Ambiguity

### Clarification Requests

```python
class ClarificationHandler:
    """
    Handle ambiguous commands by requesting clarification
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.ambiguity_patterns = [
            r"there.*", r"it.*", r"that.*",  # Unclear references
            r"over there", r"around here",   # Unclear locations
            r"something", r"thing",          # Unclear objects
        ]

    def detect_ambiguity(self, command: str) -> bool:
        """
        Detect if a command contains ambiguous elements
        """
        import re
        command_lower = command.lower()

        for pattern in self.ambiguity_patterns:
            if re.search(pattern, command_lower):
                return True

        # Check for pronouns without clear referents
        pronoun_patterns = [r"\b(he|she|it|they|them|him|her)\b"]
        for pattern in pronoun_patterns:
            if re.search(pattern, command_lower):
                # Simple check - in practice, you'd use NLP to resolve references
                return True

        return False

    def generate_clarification_request(self, ambiguous_command: str) -> str:
        """
        Generate a clarification request for an ambiguous command
        """
        prompt = f"""
        The following command is ambiguous: "{ambiguous_command}"

        Identify what information is unclear and generate a polite clarification request.

        AMBIGUOUS ELEMENTS:
        - Unclear object references
        - Unclear location references
        - Unclear action specifications
        - Missing critical parameters

        Generate a clarification request that asks for the missing information.

        RESPONSE FORMAT:
        {{
            "clarification_needed": true/false,
            "unclear_elements": ["list", "of", "unclear", "elements"],
            "clarification_request": "Polite question asking for clarification"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("clarification_request", "Can you please clarify your request?")

        except Exception as e:
            return "I didn't understand your command. Can you please rephrase it?"
```

## Learning and Adaptation

### Plan Refinement Based on Experience

```python
class PlanLearningSystem:
    """
    System that learns from plan execution experiences
    """

    def __init__(self):
        self.execution_history = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.adaptation_rules = []

    def record_execution(self, command: str, plan: List[Dict], result: str, execution_time: float):
        """
        Record plan execution for learning
        """
        execution_record = {
            "command": command,
            "plan": plan,
            "result": result,
            "execution_time": execution_time,
            "timestamp": time.time()
        }

        self.execution_history.append(execution_record)

        # Update patterns based on success/failure
        if "success" in result.lower():
            self._update_success_patterns(execution_record)
        else:
            self._update_failure_patterns(execution_record)

    def _update_success_patterns(self, record: Dict):
        """
        Update patterns for successful executions
        """
        command = record["command"]
        plan = record["plan"]

        for action in plan:
            action_key = f"{action['action_type']}_{json.dumps(action.get('parameters', {}), sort_keys=True)}"

            if action_key not in self.success_patterns:
                self.success_patterns[action_key] = {"count": 0, "avg_time": 0, "contexts": []}

            pattern = self.success_patterns[action_key]
            pattern["count"] += 1
            # Update average time
            total_time = pattern["avg_time"] * (pattern["count"] - 1) + record["execution_time"]
            pattern["avg_time"] = total_time / pattern["count"]
            pattern["contexts"].append(record["command"])

    def _update_failure_patterns(self, record: Dict):
        """
        Update patterns for failed executions
        """
        command = record["command"]
        plan = record["plan"]

        for action in plan:
            action_key = f"{action['action_type']}_{json.dumps(action.get('parameters', {}), sort_keys=True)}"

            if action_key not in self.failure_patterns:
                self.failure_patterns[action_key] = {"count": 0, "failure_reasons": []}

            self.failure_patterns[action_key]["count"] += 1
            self.failure_patterns[action_key]["failure_reasons"].append(record["result"])

    def adapt_plan(self, new_command: str, proposed_plan: List[Dict]) -> List[Dict]:
        """
        Adapt a proposed plan based on historical patterns
        """
        adapted_plan = []

        for action in proposed_plan:
            adapted_action = self._adapt_action(new_command, action)
            adapted_plan.append(adapted_action)

        return adapted_plan

    def _adapt_action(self, command: str, action: Dict) -> Dict:
        """
        Adapt a single action based on learning
        """
        action_key = f"{action['action_type']}_{json.dumps(action.get('parameters', {}), sort_keys=True)}"

        # Check if this action has failed before in similar contexts
        if action_key in self.failure_patterns:
            failure_count = self.failure_patterns[action_key]["count"]
            if failure_count > 2:  # If failed more than twice
                # Modify the action or add safety measures
                action["parameters"]["safety_margin"] = 0.5  # Add extra safety
                action["description"] += " (with extra caution due to past failures)"

        # If this action has been successful, we might optimize it
        if action_key in self.success_patterns:
            success_count = self.success_patterns[action_key]["count"]
            if success_count > 5:  # If successful many times
                # We could potentially optimize parameters here
                pass

        return action
```

## Best Practices for LLM-Based Cognitive Planning

1. **Prompt Engineering**: Carefully design prompts to guide the LLM toward correct outputs
2. **Safety Validation**: Always validate LLM outputs against safety constraints
3. **Context Awareness**: Provide sufficient context for the LLM to make informed decisions
4. **Error Handling**: Implement robust error handling for both LLM failures and action execution failures
5. **Continuous Learning**: Use execution feedback to improve future planning
6. **Human-in-the-Loop**: For critical tasks, implement human oversight and approval mechanisms

## Integration with Natural Language Understanding

The cognitive planning system should work seamlessly with natural language understanding components to create a complete human-robot interaction pipeline.

## Next Steps

In the next chapter, we'll explore the capstone project that integrates all the concepts learned throughout the modules, creating a complete autonomous humanoid system with voice command interface, cognitive planning, and physical execution capabilities.