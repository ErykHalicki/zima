import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import numpy as np

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

    def __init__(self, action_chunk_size=1, action_history_size=0, action_size=2):
        '''
        action_chunk_size: number of actions to predict at each inference
        action_history_size: number of actions to use for context as input
        action_size: size of each action (eg. 2 for 2 wheels)
        '''
        super().__init__()
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        for param in resnet.parameters():
            param.requires_grad = True 

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.action_head = nn.Sequential(
            nn.Linear(512+action_history_size*action_size, 128),  # ResNet18 outputs 512 features + we input the action hitosry
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_size*action_chunk_size) 
            # Output: [B, action_chunk_size * action_size]
            #output later gets transformed into [B, action_chunk_size, action_size]
        )

        self.action_chunk_size = action_chunk_size
        self.action_size = action_size

    def forward(self, x, action_history):
        #need to transform image to 3x224x224
        resnet_features = torch.flatten(self.feature_extractor(x), start_dim=1)
        flattened_action_history = torch.flatten(action_history, start_dim=1) 
        #use start_dim=1 to avoid flattening the entire batch
        features = torch.cat((resnet_features, flattened_action_history), dim=1)
        action_chunk = self.action_head(features).view(-1, self.action_chunk_size, self.action_size)
        return action_chunk
    
    @staticmethod
    def convert_image_to_resnet(image):
        return ActionResNet.transform(image)
    
    @staticmethod
    def bin_action(action):
        '''
        Bins a continuous action [left_speed, right_speed] into 1 of 5 classes (one-hot encoded)
        Classes: 0=stop, 1=forward, 2=backward, 3=right, 4=left
        Returns: one-hot encoded vector of shape [5]
        '''
        left_speed, right_speed = action

        SPEED_THRESHOLD = 0.1
        TURNING_THRESHOLD = 0.1

        avg_speed = (left_speed + right_speed) / 2
        speed_diff = left_speed - right_speed

        # Determine class
        if abs(speed_diff) > TURNING_THRESHOLD:
            if speed_diff > 0:
                class_idx = 3  # right (left wheel faster)
            else:
                class_idx = 4  # left (right wheel faster)
        else:  # Forward/backward is dominant
            if avg_speed > SPEED_THRESHOLD:
                class_idx = 1  # forward
            elif avg_speed < -SPEED_THRESHOLD:
                class_idx = 2  # backward
            else:
                class_idx = 0  # stop

        one_hot = np.zeros(5, dtype=np.float32)
        one_hot[class_idx] = 1.0
        return one_hot

        
