import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial  

class SerialOutputNode(Node):
    def __init__(self):
        super().__init__('serial_output_node')

        opened = False
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("ttyUSB0 opened successfully.")
            opened = True
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.serial_port = None

        if opened == False :
            try:
                self.serial_port = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
                self.get_logger().info("ttyACM0 opened successfully.")
            except serial.SerialException as e:
                self.get_logger().error(f"Failed to open serial port: {e}")
                self.serial_port = None

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

    def joint_callback(self, msg):
        if not self.serial_port or not self.serial_port.is_open:
            self.get_logger().error("Serial port is not open.")
            return

        serial_msg = f'0<{msg.position[0]+90.0}\n1<{msg.position[1]+90.0}\n3<{msg.position[2]+90.0}\n4<{msg.position[3]+90.0}\n5<{msg.position[4]+90.0}\n'
                    
        #self.get_logger().info(f'message sent: {serial_msg}')

        try:
            self.serial_port.write((serial_msg).encode('utf-8'))
        except serial.SerialException as e:
            self.get_logger().error(f"Error writing to serial port: {e}")

    def destroy_node(self):
        # Close serial port if open
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SerialOutputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
