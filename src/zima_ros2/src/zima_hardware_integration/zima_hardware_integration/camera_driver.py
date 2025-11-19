#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import threading

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 10)
        self.frame_count = 0
        self.last_log_time = self.get_clock().now()

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        self.declare_parameter('output_width', -1)
        self.declare_parameter('output_height', -1)

        output_width = self.get_parameter('output_width').get_parameter_value().integer_value
        output_height = self.get_parameter('output_height').get_parameter_value().integer_value

        if (output_width > 0 and output_height <= 0) or (output_width <= 0 and output_height > 0):
            self.get_logger().error('Both output_width and output_height must be provided if one is set')
            raise ValueError('Both output_width and output_height must be provided')

        if output_width > 0 and output_height > 0:
            self.output_size = (output_width, output_height)
            self.get_logger().info(f'Image downscaling enabled: {output_width}x{output_height}')
        else:
            self.output_size = None
            self.get_logger().info('Image downscaling disabled')
        
        # Open camera with V4L2 backend (skip GStreamer)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Force MJPEG format for 30fps
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        
        # Disable buffering - read latest frame only
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check actual FPS
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(f'Camera FPS setting: {actual_fps}')
        
        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera')
            return
        
        self.get_logger().info('Camera opened, publishing to /camera/image_raw/compressed')
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Publish on timer for stable rate
        self.timer = self.create_timer(0.033, self.publish_frame)  # ~30 Hz
    
    def capture_loop(self):
        """Dedicated thread for capturing frames as fast as possible"""
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
    
    def downscale_image(self, frame):
        if self.output_size is not None:
            return cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)
        return frame

    def publish_frame(self):
        """Timer callback to publish latest frame"""
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        frame = self.downscale_image(frame)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return
        
        # Create compressed image message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = jpeg.tobytes()
        
        self.publisher_.publish(msg)
        
        # Log FPS every second
        self.frame_count += 1
        current_time = self.get_clock().now()
        elapsed = (current_time - self.last_log_time).nanoseconds / 1e9
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.get_logger().debug(f'Publishing at: {fps:.2f} fps')
            self.frame_count = 0
            self.last_log_time = current_time
    
    def __del__(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
