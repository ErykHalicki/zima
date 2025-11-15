from .controller import Controller
from models.action_resnet import ActionResNet
import torch

class NNController(Controller):
    def __init__(self, weights_file_path, action_chunk_size, action_history_size, action_size):
        super().__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = ActionResNet(action_chunk_size, action_history_size, action_size).to(self.device)
        self.model.load_state_dict(torch.load(weights_file_path, weights_only=True))

        # Action normalization statistics
        self.action_mean = torch.tensor([0.428, 0.325]).to(self.device)
        self.action_std = torch.tensor([0.500, 0.579]).to(self.device)
   
    def update(self, mjmodel, mjdata, input_image, action_history):
        input_image = ActionResNet.convert_image_to_resnet(input_image).to(self.device)
        input_batch = torch.unsqueeze(input_image, 0)
        action_history_tensor = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_chunk = self.model(input_batch, action_history_tensor)

        action_chunk = torch.argmax(action_chunk.detach().cpu(), dim=2)
        print(action_chunk)
        next_action = action_chunk[0][0].item()  # Get first action from chunk

        # Execute action based on class: 0=stop, 1=forward, 2=backward, 3=right, 4=left
        if next_action == 0:
            self.stop()
        elif next_action == 1:
            self.forward()
        elif next_action == 2:
            self.backward()
        elif next_action == 3:
            self.turn_right()
        elif next_action == 4:
            self.turn_left()

        super().update(mjmodel, mjdata)

