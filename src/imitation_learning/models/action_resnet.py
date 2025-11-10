import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

class ActionResNet(nn.Module):
    transform = v2.Compose([
            v2.ToImage(), # numpy [H,W,C] -> tensor [C,H,W]
            v2.ToDtype(torch.float32, scale=True),  # convert to float and normalize to [0,1]
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # standardize using ImageNet stats
            # (0,1) -> (-2,2)
            # helps gradient flow if centered around 0
        ])

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        #for param in resnet.parameters():
            #param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.action_head = nn.Sequential(
            nn.Linear(512, 128),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Output: left + right wheel velocity
        )

    def forward(self, x):
        #need to transform image to 224x224
        features = torch.flatten(self.feature_extractor(x), start_dim=1)
        actions = self.action_head(features)
        return actions
    
    @staticmethod
    def convert_image_to_resnet(image):
        return ActionResNet.transform(image)
        
