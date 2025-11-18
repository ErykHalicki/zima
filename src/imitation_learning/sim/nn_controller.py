from .controller import Controller
from models.action_resnet import ActionResNet
import torch

class NNController(Controller):
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

        # Extract metadata
        metadata = checkpoint['metadata']
        self.action_chunk_size = metadata['action_chunk_size']
        self.action_history_size = metadata['action_history_size']
        self.action_size = metadata['action_size']

        # Print model information
        print(f"\n=== Loading Model ===")
        print(f"Model: {metadata['model_architecture']}")
        print(f"Training Date: {metadata['training_date']}")
        print(f"Total Parameters: {metadata['total_params']:,}")
        print(f"Trainable Parameters: {metadata['trainable_params']:,}")
        print(f"Dataset Size: {metadata['total_dataset_size']:,} samples")
        print(f"Action Chunk Size: {self.action_chunk_size}")
        print(f"Action History Size: {self.action_history_size}")
        print(f"Action Size: {self.action_size}")
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_test_loss' in checkpoint:
            print(f"Best Test Loss: {checkpoint['best_test_loss']:.4f}")
        elif 'last_test_loss' in checkpoint:
            print(f"Last Test Loss: {checkpoint['last_test_loss']:.4f}")
        print(f"====================\n")

        # Create and load model
        self.model = ActionResNet(self.action_chunk_size, self.action_history_size, self.action_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
   
    def update(self, mjmodel, mjdata, input_image, action_history):
        input_image = ActionResNet.convert_image_to_resnet(input_image).to(self.device)
        input_batch = torch.unsqueeze(input_image, 0)
        action_history_tensor = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_chunk = self.model(input_batch, action_history_tensor)
        sm = torch.nn.Softmax(dim=2)
        action_probs = sm(action_chunk)
        #print(f"Action chunk probabilities:\n {torch.round(action_probs, decimals=2)}")

        # Sample action from softmax distribution instead of taking argmax
        next_action = torch.multinomial(action_probs[0][0], num_samples=1).item()
        action_prob = action_probs[0][0][next_action].item()
        print(f"Executing action: {next_action} with probability: {action_prob:.3f}")

        if next_action == 0:
            self.stop()
        elif next_action == 1:
            self.forward()
        elif next_action == 2:
            self.turn_right()
        elif next_action == 3:
            self.turn_left()

        super().update(mjmodel, mjdata)

