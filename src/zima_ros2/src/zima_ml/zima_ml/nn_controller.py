from zima_ml.models.action_resnet import ActionResNet
from zima_ml.controller_base import ControllerBase
import torch
import rclpy
import numpy as np

class NNController(ControllerBase):
    def __init__(self):
        super().__init__('nn_controller')

        self.action_history_buffer = []
        self.processing_image = False

        self.declare_parameter('model_path', '')
        self.declare_parameter('action_chunk_usage_ratio', 1.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.action_chunk_usage_ratio = self.get_parameter('action_chunk_usage_ratio').get_parameter_value().double_value

        if not model_path:
            self.get_logger().error('model_path parameter is required but not set')
            raise ValueError('model_path parameter must be set')

        self.get_logger().info(f'Loading model from: {model_path}')
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)

        metadata = checkpoint['metadata']
        self.action_chunk_size = metadata['action_chunk_size']
        self.action_history_size = metadata['action_history_size']
        self.action_size = metadata['action_size']
        self.image_history_size = metadata.get('image_history_size', 0)

        self.get_logger().info("=== Loading Model ===")
        self.get_logger().info(f"Model: {metadata['model_architecture']}")
        self.get_logger().info(f"Training Date: {metadata['training_date']}")
        self.get_logger().info(f"Total Parameters: {metadata['total_params']:,}")
        self.get_logger().info(f"Trainable Parameters: {metadata['trainable_params']:,}")
        self.get_logger().info(f"Dataset Size: {metadata['total_dataset_size']:,} samples")
        self.get_logger().info(f"Action Chunk Size: {self.action_chunk_size}")
        self.get_logger().info(f"Action History Size: {self.action_history_size}")
        self.get_logger().info(f"Action Size: {self.action_size}")
        self.get_logger().info(f"Image History Size: {self.image_history_size}")
        self.get_logger().info(f"Action Chunk Usage Ratio: {self.action_chunk_usage_ratio}")
        if 'epoch' in checkpoint:
            self.get_logger().info(f"Epoch: {checkpoint['epoch']}")
        if 'best_test_loss' in checkpoint:
            self.get_logger().info(f"Best Test Loss: {checkpoint['best_test_loss']:.4f}")
        elif 'last_test_loss' in checkpoint:
            self.get_logger().info(f"Last Test Loss: {checkpoint['last_test_loss']:.4f}")
        self.get_logger().info("====================")

        self.model = ActionResNet(self.action_chunk_size, self.action_history_size, self.action_size, self.image_history_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.image_history_buffer = []
        self.action_chunk_buffer = None
        self.action_chunk_index = 0

    def camera_callback(self, msg):
        if self.processing_image:
            return

        self.processing_image = True

        rgb_array = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')

        if len(self.action_history_buffer) < self.action_history_size:
            padding_needed = self.action_history_size - len(self.action_history_buffer)
            action_history = np.zeros((padding_needed, self.action_size))
            if len(self.action_history_buffer) > 0:
                action_history = np.concatenate([action_history, np.array(self.action_history_buffer)])
        else:
            action_history = np.array(self.action_history_buffer)

        self.update(rgb_array, action_history)

        self.processing_image = False

    def update(self, input_image, action_history):
        self.image_history_buffer.append(input_image)
        if len(self.image_history_buffer) > self.image_history_size + 1:
            self.image_history_buffer.pop(0)

        chunk_usage_steps = max(1, int(self.action_chunk_size * self.action_chunk_usage_ratio))

        if self.action_chunk_buffer is None or self.action_chunk_index >= chunk_usage_steps:
            if len(self.image_history_buffer) < self.image_history_size + 1:
                padding_needed = self.image_history_size + 1 - len(self.image_history_buffer)
                first_image = self.image_history_buffer[0]
                image_stack = [first_image] * padding_needed + self.image_history_buffer
            else:
                image_stack = self.image_history_buffer

            transformed_images = [ActionResNet.convert_image_to_resnet(img).to(self.device) for img in image_stack]
            input_batch = torch.stack(transformed_images).unsqueeze(0)

            action_history_tensor = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_chunk = self.model(input_batch, action_history_tensor)
            sm = torch.nn.Softmax(dim=2)
            action_probs = sm(action_chunk)

            self.action_chunk_buffer = action_probs[0]
            self.action_chunk_index = 0

        next_action = torch.multinomial(self.action_chunk_buffer[self.action_chunk_index], num_samples=1).item()
        action_prob = self.action_chunk_buffer[self.action_chunk_index][next_action].item()

        self.get_logger().info(f"Executing action: {next_action} (chunk step {self.action_chunk_index}/{chunk_usage_steps}) with probability: {action_prob:.3f}")

        if next_action == 0:
            self.stop()
        elif next_action == 1:
            self.forward()
        elif next_action == 2:
            self.turn_right()
        elif next_action == 3:
            self.turn_left()

        executed_action = np.zeros(self.action_size, dtype=np.float32)
        executed_action[next_action] = 1.0

        self.action_history_buffer.append(executed_action)
        if len(self.action_history_buffer) > self.action_history_size:
            self.action_history_buffer.pop(0)

        self.action_chunk_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = NNController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

