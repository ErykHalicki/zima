import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import String


class JoyTesterNode(Node):
    def __init__(self):
        super().__init__('joy_tester_node')
        
        # Subscribe to joy topic
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # Publisher for human-readable output
        self.output_pub = self.create_publisher(
            String,
            'joy_test_output',
            10
        )
        
        # Keep track of last button states to detect changes
        self.last_buttons = None
        
        self.get_logger().info('Joy tester node initialized')
        self.get_logger().info('Listening for controller input on /joy topic')
        self.get_logger().info('Press buttons or move axes to see their indices')
    
    def joy_callback(self, msg):
        output = String()
        output_parts = []
        
        # Check for button presses
        if self.last_buttons is not None:
            for i, (last, current) in enumerate(zip(self.last_buttons, msg.buttons)):
                if last == 0 and current == 1:
                    button_info = f"Button {i} PRESSED"
                    self.get_logger().info(button_info)
                    output_parts.append(button_info)
                elif last == 1 and current == 0:
                    button_info = f"Button {i} RELEASED"
                    self.get_logger().info(button_info)
                    output_parts.append(button_info)
        
        # Check for axis movement (if value exceeds 0.5 in either direction)
        for i, value in enumerate(msg.axes):
            if abs(value) > 0.5:
                axis_info = f"Axis {i}: {value:.2f}"
                self.get_logger().info(axis_info)
                output_parts.append(axis_info)
        
        # Save current button states for next comparison
        self.last_buttons = list(msg.buttons)
        
        # Publish output if there's anything to report
        if output_parts:
            output.data = '\n'.join(output_parts)
            self.output_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = JoyTesterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()