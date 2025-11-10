from .controller import Controller
from models.action_resnet import ActionResNet
import torch

class NNController(Controller):
    def __init__(self, weights_file_path):
        super().__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = ActionResNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_file_path, weights_only=True))
   
    def update(self, mjmodel, mjdata, input_image):
        #expecting rgb input image
        input_image = ActionResNet.convert_image_to_resnet(input_image).to(self.device)
        input_batch = torch.unsqueeze(input_image, 0)
        actions = self.model(input_batch).clone().detach().cpu()
        print(actions)
        self.left_speed = actions[0][0]*self.max_speed
        self.right_speed = actions[0][1]*self.max_speed
        super().update(mjmodel, mjdata)

