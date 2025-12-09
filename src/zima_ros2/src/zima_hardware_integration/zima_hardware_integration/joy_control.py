import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from zima_msgs.msg import MotorCommand, ArmStateDelta, ServoCommand
from geometry_msgs.msg import Vector3
from std_msgs.msg import Empty

class JoyControlNode(Node):
    def __init__(self):
        super().__init__('joy_control_node')

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

        self.AXIS_LEFT_X = 0
        self.AXIS_LEFT_Y = 1
        self.AXIS_RIGHT_X = 3
        self.AXIS_RIGHT_Y = 4

        self.declare_parameter('joy_deadzone', 0.05)
        self.declare_parameter('max_speed', 255)

        self.joy_deadzone = self.get_parameter('joy_deadzone').value
        self.max_speed = self.get_parameter('max_speed').value

        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.motor_pub = self.create_publisher(MotorCommand, '/motor_command', 10)
        self.arm_delta_pub = self.create_publisher(ArmStateDelta, '/arm_state_delta', 10)
        self.servo_pub = self.create_publisher(ServoCommand, '/servo_command', 10)
        self.reset_arm_pub = self.create_publisher(Empty, '/reset_arm', 10)

        self.get_logger().info('Joy control node initialized')

    def joy_callback(self, msg):
        if msg.buttons[self.BTN_R1]:
            self.emergency_stop()
            return

        self.process_track_commands(msg)
        self.process_arm_commands(msg)

    def process_track_commands(self, msg):
        linear = 0.0
        angular = 0.0

        if msg.buttons[self.BTN_DPAD_UP]:
            linear = 1.0
        elif msg.buttons[self.BTN_DPAD_DOWN]:
            linear = -1.0

        if msg.buttons[self.BTN_DPAD_LEFT]:
            angular = 1.0
        elif msg.buttons[self.BTN_DPAD_RIGHT]:
            angular = -1.0

        left_speed = linear - angular
        right_speed = linear + angular

        max_value = max(abs(left_speed), abs(right_speed), 1.0)
        left_speed = left_speed / max_value
        right_speed = right_speed / max_value

        left_speed_scaled = int(left_speed * self.max_speed)
        right_speed_scaled = int(right_speed * self.max_speed)

        self.send_motor_command(MotorCommand.MOTOR_LEFT, left_speed_scaled)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, right_speed_scaled)

    def process_arm_commands(self, msg):
        left_x = self.apply_deadzone(msg.axes[self.AXIS_LEFT_X])
        left_y = self.apply_deadzone(msg.axes[self.AXIS_LEFT_Y])
        right_x = self.apply_deadzone(msg.axes[self.AXIS_RIGHT_X])
        right_y = self.apply_deadzone(msg.axes[self.AXIS_RIGHT_Y])

        gripper = 0.0
        if msg.buttons[self.BTN_L2]:
            gripper -= 1.0
        if msg.buttons[self.BTN_R2]:
            gripper += 1.0

        arm_delta = ArmStateDelta()
        arm_delta.translation = Vector3(x=left_y, y=left_x, z=right_y)
        arm_delta.orientation = Vector3(x=-right_x, y=0.0, z=0.0)
        arm_delta.gripper = gripper

        self.arm_delta_pub.publish(arm_delta)

    def apply_deadzone(self, value):
        if abs(value) < self.joy_deadzone:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        return sign * (abs(value) - self.joy_deadzone) / (1.0 - self.joy_deadzone)

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

    def emergency_stop(self):
        self.get_logger().info('Emergency stop activated', throttle_duration_sec=1.0)

        for servo_id in range(8):
            command = ServoCommand()
            command.header.stamp = self.get_clock().now().to_msg()
            command.servo_id = servo_id
            if servo_id == 3:
                command.position = 181.0
            else:
                command.position = -1.0
            self.servo_pub.publish(command)

        self.reset_arm_pub.publish(Empty())

        self.send_motor_command(MotorCommand.MOTOR_LEFT, 0)
        self.send_motor_command(MotorCommand.MOTOR_RIGHT, 0)

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
