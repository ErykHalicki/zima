from .controller import Controller
from models.action_resnet import ActionResNet
import torch

class NNController(Controller):
    action_chunk_usage_ratio = 0.0

    def __init__(self, model_path):
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"


        # Load checkpoint with metadata
        checkpoint = torch.load(model_path, weights_only=False)

        metadata = checkpoint['metadata']
        self.action_chunk_size = metadata['action_chunk_size']
        self.action_history_size = metadata['action_history_size']
        self.action_size = metadata['action_size']
        self.image_history_size = metadata.get('image_history_size', 0)

        print(f"\n=== Loading Model ===")
        print(f"Model: {metadata['model_architecture']}")
        print(f"Training Date: {metadata['training_date']}")
        print(f"Total Parameters: {metadata['total_params']:,}")
        print(f"Trainable Parameters: {metadata['trainable_params']:,}")
        print(f"Dataset Size: {metadata['total_dataset_size']:,} samples")
        print(f"Action Chunk Size: {self.action_chunk_size}")
        print(f"Action History Size: {self.action_history_size}")
        print(f"Action Size: {self.action_size}")
        print(f"Image History Size: {self.image_history_size}")
        print(f"Action Chunk Usage Ratio: {self.action_chunk_usage_ratio}")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_test_loss' in checkpoint:
            print(f"Best Test Loss: {checkpoint['best_test_loss']:.4f}")
        elif 'last_test_loss' in checkpoint:
            print(f"Last Test Loss: {checkpoint['last_test_loss']:.4f}")
        print(f"====================\n")

        self.model = ActionResNet(self.action_chunk_size, self.action_history_size, self.action_size, self.image_history_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.image_history_buffer = []
        self.action_chunk_buffer = None
        self.action_chunk_index = 0

    def update(self, mjmodel, mjdata, input_image, action_history):
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
        print(f"Executing action: {next_action} (chunk step {self.action_chunk_index}/{chunk_usage_steps}) with probability: {action_prob:.3f}")

        if next_action == 0:
            self.stop()
        elif next_action == 1:
            self.forward()
        elif next_action == 2:
            self.turn_right()
        elif next_action == 3:
            self.turn_left()

        self.action_chunk_index += 1

        super().update(mjmodel, mjdata)

