'''
import torch
from torchvision import transforms
from PIL import Image
pil_image = Image.fromarray(numpy_image)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])

class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, images, actions, transform=None):
        self.images = images  # List of PIL Images or numpy arrays
        self.actions = actions
        self.transform = transform
    
    def __getitem__(self, idx):
        image = self.images[idx]
        action = self.actions[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, action
    
    def __len__(self):
        return len(self.images)

# Usage
dataset = RobotDataset(images, actions, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
'''
