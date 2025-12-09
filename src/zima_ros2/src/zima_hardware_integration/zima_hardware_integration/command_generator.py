import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from zima_msgs.msg import HardwareCommand, ServoCommand, MotorCommand

class CommandGeneratorNode(Node):
    def __init__(self):
        super().__init__('command_generator_node')
        
        # Subscribe to high-level control topics
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/goal_joint_state',
            self.joint_state_callback,
            10
        )
        
        self.servo_command_sub = self.create_subscription(
            ServoCommand,
            '/servo_command',
            self.servo_command_callback,
            10
        )
        
        self.motor_command_sub = self.create_subscription(
            MotorCommand,
            '/motor_command',
            self.motor_command_callback,
            10
        )
        
        # Publisher for hardware commands
        self.hw_command_pub = self.create_publisher(
            HardwareCommand,
            'hardware_command',
            10
        )
        
        self.get_logger().info('Command generator node initialized')
    
    def joint_state_callback(self, msg): #interface for arm controller
        """
        Convert JointState messages to hardware commands for servos.
        expects all values to be in range 0-180 degrees
        """
        servo_indices = [0, 1, 3, 4, 5] #base, shoulder, elbow, hand, gripper 

        if len(msg.position) < len(servo_indices):
            self.get_logger().error("JointState message has insufficient position data")
            return

        for position in msg.position:
            if position > 180 or position < 0:
                self.get_logger().error("JointState message has out of bounds position data")
                return

        if len(servo_indices) != len(msg.position):
            raise Exception(f"Size of provided joint state ({len(msg.position)}) != {len(servo_indices)}")
        
        # Create and publish a command for each servo
        for i, servo_idx in enumerate(servo_indices):
            command = HardwareCommand()
            command.header = msg.header
            command.subsystem = servo_idx  # Servo ID (0-7)
            command.value = msg.position[i]   

            self.hw_command_pub.publish(command)
    
    def servo_command_callback(self, msg): # only for manual testing of servos, do not use in arm controller
        """Convert ServoCommand messages to hardware commands."""
        if msg.servo_id > 7:
            self.get_logger().error(f"Invalid servo ID: {msg.servo_id}. Must be 0-7.")
            return
        
        command = HardwareCommand()
        command.header = msg.header
        command.subsystem = msg.servo_id
        command.value = msg.position
        
        self.hw_command_pub.publish(command)
    
    def motor_command_callback(self, msg):
        """Convert MotorCommand messages to hardware commands."""
        command = HardwareCommand()
        command.header = msg.header
        
        # Map motor ID and action to subsystem ID
        if msg.motor_id == msg.MOTOR_LEFT:
            if msg.action == msg.ACTION_FORWARD:
                command.subsystem = 8  # Left motor forward
            elif msg.action == msg.ACTION_REVERSE:
                command.subsystem = 9  # Left motor reverse
            elif msg.action == msg.ACTION_STOP:
                command.subsystem = 10  # Left motor stop
            else:
                self.get_logger().error(f"Invalid motor action: {msg.action}")
                return
        elif msg.motor_id == msg.MOTOR_RIGHT:
            if msg.action == msg.ACTION_FORWARD:
                command.subsystem = 12  # Right motor reverse - motor inverted irl
            elif msg.action == msg.ACTION_REVERSE:
                command.subsystem = 11  # Right motor forward
            elif msg.action == msg.ACTION_STOP:
                command.subsystem = 13  # Right motor stop
            else:
                self.get_logger().error(f"Invalid motor action: {msg.action}")
                return
        else:
            self.get_logger().error(f"Invalid motor ID: {msg.motor_id}")
            return
        
        command.value = float(msg.speed)
        
        self.hw_command_pub.publish(command)

def main(args=None):
    rclpy.init(args=args)
    node = CommandGeneratorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
