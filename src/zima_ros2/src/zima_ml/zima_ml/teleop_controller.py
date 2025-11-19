from zima_ml.controller_base import ControllerBase
from zima_ml.teleop_server import start_server, get_control_state, set_camera_frame
import rclpy
from threading import Thread

class TeleopController(ControllerBase):
    def __init__(self):
        super().__init__('teleop_controller')

        self.declare_parameter('update_rate', 50.0)
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value

        self.server_thread = Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.get_logger().info(f'Teleop server started on http://0.0.0.0:5000')
        self.get_logger().info(f'Control update rate: {update_rate} Hz')

        self.timer = self.create_timer(1.0 / update_rate, self.control_loop)

    def camera_callback(self, msg):
        rgb_array = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
        set_camera_frame(rgb_array)

    def control_loop(self):
        state = get_control_state()

        forward = state['forward']
        backward = state['backward']
        left = state['left']
        right = state['right']

        if forward > 0 and backward == 0 and left == 0 and right == 0:
            self.forward()
        elif backward > 0 and forward == 0 and left == 0 and right == 0:
            self.backward()
        elif left > 0 and forward == 0 and backward == 0 and right == 0:
            self.turn_left()
        elif right > 0 and forward == 0 and backward == 0 and left == 0:
            self.turn_right()
        else:
            self.stop()

def main(args=None):
    rclpy.init(args=args)
    node = TeleopController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
