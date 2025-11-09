import torch
import torch.nn as nn
import torchvision

class ActionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.action_head = nn.Sequential(
            nn.Linear(512, 128),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(128, 2) # Output: left + right wheel velocity
        )

    def forward(self, x):
        #need to transform image to 224x224
        features = nn.Flatten(self.feature_extractor(x))
        actions = self.action_head(features)
        return actions


