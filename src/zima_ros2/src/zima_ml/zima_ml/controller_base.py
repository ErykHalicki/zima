import rclpy
from rclpy.node import Node
from zima_msgs.msg import MotorCommand
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class ControllerBase(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.motor_pub = self.create_publisher(MotorCommand, 'motor_command', 10)
        self.camera_sub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_callback, 10)

        self.bridge = CvBridge()

        self.declare_parameter('max_speed', 255)
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().integer_value

    def camera_callback(self, msg):
        pass

    def send_motor_command(self, motor_id, speed):
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

    def stop(self):
        self.send_motor_command(MotorCommand.MOTOR_LEFT, 0)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, 0)

    def forward(self):
        self.send_motor_command(MotorCommand.MOTOR_LEFT, self.max_speed)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, self.max_speed)

    def backward(self):
        self.send_motor_command(MotorCommand.MOTOR_LEFT, -self.max_speed)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, -self.max_speed)

    def turn_right(self):
        self.send_motor_command(MotorCommand.MOTOR_LEFT, self.max_speed)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, -self.max_speed)

    def turn_left(self):
        self.send_motor_command(MotorCommand.MOTOR_LEFT, -self.max_speed)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, self.max_speed)
