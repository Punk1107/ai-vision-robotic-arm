"""
ROS 2 Integration Node for AI Robotic Arm
=========================================
This node bridges the ROS 2 ecosystem with the custom Python RobotController.
It subscribes to a topic for target coordinates or joint angles and sends
the commands to the Arduino via the RobotController.
"""

import os
import sys

# Add the project root to the python path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Point
    import json
except ImportError:
    print("ROS 2 (rclpy) is not installed. Please source your ROS 2 installation.")
    print("Example: source /opt/ros/humble/setup.bash")
    sys.exit(1)

from src.robotics.control import RobotController
from src.robotics.kinematics import IKSolver

class ArmControlNode(Node):
    def __init__(self):
        super().__init__('arm_control_node')
        
        # Initialize IK and Hardware Controller
        self.get_logger().info("Initializing IK Solver and Robot Controller...")
        self.ik_solver = IKSolver(elbow_up=True)
        self.controller = RobotController()
        
        # Open serial connection
        self.controller.connect()
        self.controller.home()
        
        # Subscribe to target point (XYZ in meters)
        self.subscription = self.create_subscription(
            Point,
            '/vision/target',
            self.target_callback,
            10
        )
        
        # Command subscription (JSON string for direct hardware control)
        self.cmd_subscription = self.create_subscription(
            String,
            '/arm/command',
            self.command_callback,
            10
        )
        
        self.get_logger().info("Arm Control Node is ready and listening.")

    def target_callback(self, msg):
        """
        Receive an XYZ point, solve IK, and move the arm.
        """
        self.get_logger().info(f"Received target: X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
        import numpy as np
        target_xyz = np.array([msg.x, msg.y, msg.z])
        
        try:
            angles = self.ik_solver.solve(target_xyz)
            self.get_logger().info(f"Solved IK: {angles}")
            self.controller.move_to(angles, blocking=False)
        except Exception as e:
            self.get_logger().error(f"IK Solver failed: {e}")

    def command_callback(self, msg):
        """
        Receive a direct JSON command string and forward to the controller.
        Example: {"cmd": "grip", "value": 90}
        """
        try:
            data = json.loads(msg.data)
            self.get_logger().info(f"Received raw command: {data}")
            
            cmd = data.get("cmd")
            if cmd == "home":
                self.controller.home()
            elif cmd == "grip":
                self.controller.grip(close=(data.get("value", 0) > 45))
            elif cmd == "estop":
                self.controller.emergency_stop()
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON command received.")

    def destroy_node(self):
        # Ensure arm is homed and serial is closed on shutdown
        self.get_logger().info("Shutting down arm control node...")
        try:
            self.controller.home()
            self.controller.close()
        except:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
