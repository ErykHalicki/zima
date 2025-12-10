import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from zima_msgs.msg import HardwareStatus, EncoderData, ServoPositions
import serial
import re

class SerialReceiverNode(Node):
    def __init__(self):
        super().__init__('serial_receiver_node')
        
        # Declare parameters
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('timeout', 0.1)
        self.declare_parameter('update_rate', 200.0)  # Hz
        
        # Get parameters
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        timeout = self.get_parameter('timeout').get_parameter_value().double_value
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value
        
        # Initialize serial connection
        self.serial_port = None
        self.connect_to_serial(serial_port, baud_rate, timeout)
        
        # Create publishers
        self.hardware_status_pub = self.create_publisher(
            HardwareStatus, 
            '/hardware_status', 
            10
        )
        
        self.encoder_data_pub = self.create_publisher(
            EncoderData,
            '/encoder_data',
            10
        )
        
        # Joint state publisher for RViz visualization
        self.joint_states_pub = self.create_publisher(
            JointState,
            '/current_joint_state',
            10
        )
        
        # Create timer for periodic reading
        period = 1.0 / update_rate
        self.timer = self.create_timer(period, self.read_serial_data)
        
        self.get_logger().info('Serial receiver node initialized')
    
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
    
    def read_serial_data(self):
        """Read and process data from the serial port."""
        if not self.serial_port or not self.serial_port.is_open:
            self.get_logger().error("Serial port is not open")
            return
        
        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line.startswith('ZIMA:DATA:'):
                    self.process_hardware_data(line)
        except serial.SerialException as e:
            self.get_logger().error(f"Error reading from serial port: {e}")
        except UnicodeDecodeError as e:
            self.get_logger().error(f"Error decoding serial data: {e}")
    
    def process_hardware_data(self, data_line):
        """Process the hardware data line from the ESP32."""
        # Parse the data line
        # Format: ZIMA:DATA:CLICKSLEFT=<value>,CLICKSRIGHT=<value>,SERVOS=<pos1>|<pos2>|...|<pos8>
        
        try:
            # Remove prefix
            data_part = data_line[len('ZIMA:DATA:'):]
            
            # Extract encoder data
            left_clicks_match = re.search(r'CLICKSLEFT=(-?\d+)', data_part)
            right_clicks_match = re.search(r'CLICKSRIGHT=(-?\d+)', data_part)
            
            if left_clicks_match and right_clicks_match:
                left_clicks = int(left_clicks_match.group(1))
                right_clicks = int(right_clicks_match.group(1))
                
                # Publish encoder data
                encoder_msg = EncoderData()
                encoder_msg.header = self.create_header()
                encoder_msg.left_clicks = left_clicks
                encoder_msg.right_clicks = right_clicks
                self.encoder_data_pub.publish(encoder_msg)
            
            # Extract servo positions
            servo_match = re.search(r'SERVOS=([-\d.|]+)', data_part)
            if servo_match:
                servo_str = servo_match.group(1)
                servo_positions = [float(pos) for pos in servo_str.split('|')]
                
                self.publish_joint_states(servo_positions)
            
            # Publish complete hardware status
            status_msg = HardwareStatus()
            status_msg.header = self.create_header()
            if left_clicks_match and right_clicks_match:
                status_msg.clicks_left = left_clicks
                status_msg.clicks_right = right_clicks
            if servo_match:
                status_msg.servo_positions = servo_positions
            
            self.hardware_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing hardware data: {e}")
    
    def create_header(self):
        """Create a standard header with current time."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'zima_base'
        return header
    
    def publish_joint_states(self, servo_positions):
        """Publish joint states from servo positions."""
        # Define joint names 
        joint_names = [
            'base_joint',       # Base rotation 
            'shoulder_joint',   # Shoulder pitch
            'elbow_joint',      # Elbow pitch
            'hand_joint',       # Hand pitch
            'wrist_joint',      # Wrist rotation
        ]
        
        # Ensure we have enough servo positions
        if len(servo_positions) >= len(joint_names):
            joint_msg = JointState()
            joint_msg.header = self.create_header()
            joint_msg.name = joint_names
            
            # You might need to adjust this conversion based on your actual servo range
            joint_msg.position = [
                float(pos - 90.0) for pos in servo_positions[:len(joint_names)]
            ]
            
            # Publish joint states
            self.joint_states_pub.publish(joint_msg)#published from -90 to 90
    
    def destroy_node(self):
        """Clean up when node is shutting down."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info("Serial port closed")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialReceiverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
