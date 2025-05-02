import rclpy
from rclpy.node import Node
from zima_msgs.msg import HardwareCommand
import serial

class SerialSenderNode(Node):
    def __init__(self):
        super().__init__('serial_sender_node')
        
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 0.1)
        
        # Get parameters
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        timeout = self.get_parameter('timeout').get_parameter_value().double_value
        
        # Initialize serial connection
        self.serial_port = None
        self.connect_to_serial(serial_port, baud_rate, timeout)
        
        # Create subscription
        self.subscription = self.create_subscription(
            HardwareCommand,
            'hardware_command',
            self.command_callback,
            10
        )
        
        self.get_logger().info('Serial sender node initialized')
    
    def connect_to_serial(self, port, baud_rate, timeout):
        """Attempt to connect to the serial port."""
        # Try the specified port
        try:
            self.serial_port = serial.Serial(port, baud_rate, timeout=timeout)
            self.get_logger().info(f"Connected to serial port: {port}")
            return
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port {port}: {e}")
        
        # If the specified port failed, try common alternatives
        alternate_ports = ['/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1', '/dev/ttyACM1']
        for alt_port in alternate_ports:
            if alt_port == port:
                continue  # Skip the already tried port
            
            try:
                self.serial_port = serial.Serial(alt_port, baud_rate, timeout=timeout)
                self.get_logger().info(f"Connected to alternative serial port: {alt_port}")
                return
            except serial.SerialException:
                pass
        
        self.get_logger().error("Failed to connect to any serial port")
    
    def command_callback(self, msg):
        """Process and send hardware command to serial port."""
        if not self.serial_port or not self.serial_port.is_open:
            self.get_logger().error("Serial port is not open")
            return
        
        try:
            # Format command according to the hardware interface protocol
            # Format: subsystem<value
            command = f"{msg.subsystem}<{msg.value}\n"
            
            # Log command if debug is enabled
            self.get_logger().debug(f"Sending command: {command.strip()}")
            
            # Send command
            self.serial_port.write(command.encode('utf-8'))
            
        except serial.SerialException as e:
            self.get_logger().error(f"Error writing to serial port: {e}")
    
    def destroy_node(self):
        """Clean up when node is shutting down."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info("Serial port closed")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialSenderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()