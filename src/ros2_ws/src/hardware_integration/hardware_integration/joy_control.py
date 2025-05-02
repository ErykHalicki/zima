import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from zima_msgs.msg import MotorCommand, ServoCommand
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class JoyControlNode(Node):
    def __init__(self):
        super().__init__('joy_control_node')
        
        # PS3 controller button/axis mapping constants
        # Buttons
        self.BTN_X = 0
        self.BTN_CIRCLE = 1
        self.BTN_TRIANGLE = 2
        self.BTN_SQUARE = 3
        self.BTN_L1 = 4
        self.BTN_R1 = 5
        self.BTN_L2 = 6
        self.BTN_R2 = 7
        self.BTN_SELECT = 8
        self.BTN_START = 9
        self.BTN_PS = 10
        self.BTN_L3 = 11
        self.BTN_R3 = 12
        self.BTN_DPAD_UP = 13
        self.BTN_DPAD_DOWN = 14
        self.BTN_DPAD_LEFT = 15
        self.BTN_DPAD_RIGHT = 16
        
        # Axes
        self.AXIS_LEFT_X = 1   # Left stick left/right
        self.AXIS_LEFT_Y = 0   # Left stick up/down
        self.AXIS_RIGHT_X = 3  # Right stick left/right
        self.AXIS_RIGHT_Y = 4  # Right stick up/down
        self.AXIS_L2 = 2       # L2 trigger (-1 is pressed, 1 is up)
        self.AXIS_R2 = 5       # R2 trigger (-1 is pressed, 1 is up)
        
        # Define parameters
        self.declare_parameter(
            'joy_deadzone', 
            0.05, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Deadzone for joystick inputs (0.0 to 1.0)'
            )
        )
        self.declare_parameter(
            'max_speed', 
            255, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Maximum speed value (0-255)'
            )
        )
        self.declare_parameter(
            'linear_axis', 
            self.AXIS_LEFT_Y, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Joystick axis for linear motion (forward/backward)'
            )
        )
        self.declare_parameter(
            'angular_axis', 
            self.AXIS_LEFT_X, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Joystick axis for angular motion (left/right)'
            )
        )
        self.declare_parameter(
            'enable_button', 
            self.BTN_R1, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Button to enable motor commands (R1 on PS3 controller)'
            )
        )
        self.declare_parameter(
            'emergency_stop_button', 
            self.BTN_R2, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Button to stop all motors (R2 on PS3 controller)'
            )
        )

        # Get parameter values
        self.joy_deadzone = self.get_parameter('joy_deadzone').value
        self.max_speed = self.get_parameter('max_speed').value
        self.linear_axis = self.get_parameter('linear_axis').value
        self.angular_axis = self.get_parameter('angular_axis').value
        self.enable_button = self.get_parameter('enable_button').value
        self.emergency_stop_button = self.get_parameter('emergency_stop_button').value
        
        # Control mode
        self.control_mode = "drive"  # default mode
        
        # Create subscriptions
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # Create publishers
        self.motor_pub = self.create_publisher(
            MotorCommand,
            '/motor_command',
            10
        )
        
        self.servo_pub = self.create_publisher(
            ServoCommand,
            '/servo_command',
            10
        )
        
        # For status messages, we can avoid spamming the log
        self.timer = self.create_timer(5.0, self.status_timer_callback)
        self.get_logger().info('Joy control node initialized')
        
    def status_timer_callback(self):
        """Periodic status message"""
        self.get_logger().info(f'Joy control active. Current mode: {self.control_mode}')
        
    def joy_callback(self, msg):
        """Process joystick inputs and generate commands"""
        # Check if we have enough axes and buttons in the message
        if len(msg.axes) <= max(self.linear_axis, self.angular_axis) or \
           len(msg.buttons) <= max(self.enable_button, self.emergency_stop_button):
            self.get_logger().warn('Joy message has insufficient data')
            return
            
        # Check for mode switching
        if msg.buttons[self.BTN_SELECT] == 1:
            self.cycle_control_mode()
            return
            
        # Check if emergency stop button is pressed
        if msg.buttons[self.emergency_stop_button] == 1:
            self.emergency_stop()
            return
        
        # Process based on current control mode
        if self.control_mode == "drive":
            self.process_drive_commands(msg)
        elif self.control_mode == "arm":
            self.process_arm_commands(msg)
        elif self.control_mode == "custom1":
            self.process_custom1_commands(msg)
        elif self.control_mode == "custom2":
            self.process_custom2_commands(msg)
    
    def process_drive_commands(self, msg):
        """Process commands in drive mode"""
        # Only process commands if enable button is pressed
        if msg.buttons[self.enable_button] != 1:
            return
            
        # Get joystick values
        linear = msg.axes[self.linear_axis]
        angular = msg.axes[self.angular_axis]
        
        # Apply deadzone
        linear = self.apply_deadzone(linear)
        angular = self.apply_deadzone(angular)
        
        # Calculate left and right motor values using differential drive calculation
        left_speed = linear + angular
        right_speed = linear - angular
        
        # Normalize speeds to ensure they stay within bounds (-1.0 to 1.0)
        max_value = max(abs(left_speed), abs(right_speed), 1.0)
        left_speed = left_speed / max_value
        right_speed = right_speed / max_value
        
        # Scale to max_speed
        left_speed_scaled = int(left_speed * self.max_speed)
        right_speed_scaled = int(right_speed * self.max_speed)
        
        # Send motor commands
        self.send_motor_command(MotorCommand.MOTOR_LEFT, left_speed_scaled)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, right_speed_scaled)
    
    def process_arm_commands(self, msg):
        """Process commands in arm control mode"""
        # Empty function for future arm control implementation
        pass
    
    def process_custom1_commands(self, msg):
        """Process commands in custom mode 1"""
        # Empty function for future custom mode implementation
        pass
    
    def process_custom2_commands(self, msg):
        """Process commands in custom mode 2"""
        # Empty function for future custom mode implementation
        pass
    
    def handle_dpad_inputs(self, msg):
        """Handle D-pad inputs for various functions"""
        # Empty function for future D-pad control implementation
        pass
    
    def handle_shape_buttons(self, msg):
        """Handle X, Circle, Triangle, Square buttons"""
        # Empty function for future shape button implementation
        pass
    
    def handle_trigger_buttons(self, msg):
        """Handle L1, L2, R1, R2 buttons for secondary functions"""
        # Empty function for future trigger button implementation
        pass
    
    def handle_thumbstick_buttons(self, msg):
        """Handle L3 and R3 (thumbstick press) buttons"""
        # Empty function for future thumbstick button implementation
        pass
    
    def cycle_control_mode(self):
        """Cycle through different control modes"""
        modes = ["drive", "arm", "custom1", "custom2"]
        current_index = modes.index(self.control_mode)
        next_index = (current_index + 1) % len(modes)
        self.control_mode = modes[next_index]
        self.get_logger().info(f'Switching to {self.control_mode} control mode')
    
    def apply_deadzone(self, value):
        """Apply deadzone to joystick value"""
        if abs(value) < self.joy_deadzone:
            return 0.0
        
        # Scale the value from deadzone to 1.0 (or -1.0) to maintain full range
        sign = 1.0 if value > 0.0 else -1.0
        return sign * (abs(value) - self.joy_deadzone) / (1.0 - self.joy_deadzone)
    
    def send_motor_command(self, motor_id, speed):
        """Create and publish a motor command message"""
        command = MotorCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.motor_id = motor_id
        
        if speed == 0:
            command.action = MotorCommand.ACTION_STOP
            command.speed = 0
        elif speed > 0:
            command.action = MotorCommand.ACTION_FORWARD
            command.speed = abs(speed)
        else:
            command.action = MotorCommand.ACTION_REVERSE
            command.speed = abs(speed)
        
        self.motor_pub.publish(command)
    
    def send_servo_command(self, servo_id, position):
        """Create and publish a servo command message"""
        command = ServoCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.servo_id = servo_id
        command.position = position
        self.servo_pub.publish(command)
    
    def emergency_stop(self):
        """Stop all motors immediately"""
        self.get_logger().info('Emergency stop activated')
        
        # Stop left motor
        command = MotorCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.motor_id = MotorCommand.MOTOR_LEFT
        command.action = MotorCommand.ACTION_STOP
        command.speed = 0
        self.motor_pub.publish(command)
        
        # Stop right motor
        command = MotorCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.motor_id = MotorCommand.MOTOR_RIGHT
        command.action = MotorCommand.ACTION_STOP
        command.speed = 0
        self.motor_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = JoyControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
