from .controller import Controller
from models.action_resnet import ActionResNet
import torch
import cv2

class NNController(Controller):
    def __init__(self, weights_file_path, action_chunk_size, action_history_size, action_size):
        super().__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = ActionResNet(action_chunk_size, action_history_size, action_size).to(self.device)
        self.model.load_state_dict(torch.load(weights_file_path, weights_only=True))

        # Action normalization statistics
        self.action_mean = torch.tensor([0.428, 0.325]).to(self.device)
        self.action_std = torch.tensor([0.500, 0.579]).to(self.device)
        self.model.eval()
   
    def update(self, mjmodel, mjdata, input_image, action_history):
        cv2.imshow("NN INput", input_image)
        cv2.waitKey(1)
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

        # Execute action based on class: 0=stop, 1=forward, 2=backward, 3=right, 4=left
        if next_action == 0:
            self.stop()
        elif next_action == 1:
            self.forward()
        elif next_action == 2:
            self.turn_right()
        elif next_action == 3:
            self.turn_left()

        super().update(mjmodel, mjdata)

