from zima_ml.controller_base import ControllerBase
from zima_ml.teleop_server import start_server, get_control_state, set_camera_frame, print_to_terminal
from zima_ml.zima_dataset import ZimaDataset
import rclpy
from threading import Thread
import numpy as np
import time

class TeleopController(ControllerBase):
    def __init__(self):
        super().__init__('teleop_controller')

        self.declare_parameter('update_rate', 30.0)
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value

        self.server_thread = Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.get_logger().info(f'Teleop server started on http://0.0.0.0:5000')
        self.get_logger().info(f'Control update rate: {update_rate} Hz')

        self.current_image = None
        self.episode_data = {
            "images": [],
            "actions": []
        }

        self.timer = self.create_timer(1.0 / update_rate, self.control_loop)

        self.declare_parameter('dataset_path', '')
        dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value

        if not dataset_path:
            self.get_logger().error('dataset_path parameter is required but not set')
            raise ValueError('dataset_path parameter must be set')

        self.dataset = ZimaDataset(dataset_path)
        self.dataset_path = dataset_path
        self.get_logger().info(f'Dataset initialized at: {dataset_path}')
        print_to_terminal(f'Dataset path: {dataset_path}')

    def camera_callback(self, msg):
        bgr_array = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.current_image = bgr_array
        set_camera_frame(bgr_array)

    def control_loop(self):
        state = get_control_state()

        forward = state['forward']
        backward = state['backward']
        left = state['left']
        right = state['right']

        wheel_speeds = np.array([0.0, 0.0])

        if forward > 0 and backward == 0 and left == 0 and right == 0:
            self.forward()
            wheel_speeds = np.array([1.0, 1.0])
        elif backward > 0 and forward == 0 and left == 0 and right == 0:
            self.backward()
            wheel_speeds = np.array([-1.0, -1.0])
        elif left > 0 and forward == 0 and backward == 0 and right == 0:
            self.turn_left()
            wheel_speeds = np.array([-1.0, 1.0])
        elif right > 0 and forward == 0 and backward == 0 and left == 0:
            self.turn_right()
            wheel_speeds = np.array([1.0, -1.0])
        else:
            self.stop()
            wheel_speeds = np.array([0.0, 0.0])

        if self.current_image is not None:
            self.episode_data["images"].append(self.current_image.copy())
            self.episode_data["actions"].append(wheel_speeds)

        if state['discard_episode']:
            print_to_terminal(f"Episode discarded ({len(self.episode_data['images'])} frames)")
            self.episode_data["images"].clear()
            self.episode_data["actions"].clear()
        elif state['save_episode']:
            save_start_time = time.time()
            episode_num = self.dataset.add_episode(self.episode_data)
            print_to_terminal(f"episode_{episode_num} saved with {len(self.episode_data['images'])} frames. Save time: {time.time()-save_start_time:.2f}s")
            print_to_terminal(f'Dataset path: {self.dataset_path}')
            self.episode_data["images"].clear()
            self.episode_data["actions"].clear()

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
