import torch
import torchvision
from torchvision.transforms import v2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.action_resnet import ActionResNet
from datasets.zima_torch_dataset import ZimaTorchDataset
import time
import cv2
from tqdm import tqdm

transform = v2.Compose([
    v2.ToImage(), # numpy [H,W,C] -> tensor [C,H,W]
    v2.ToDtype(torch.float32, scale=True),  # convert to float and normalize to [0,1]
    v2.Resize((224, 224), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # standardize using ImageNet stats
    # (0,1) -> (-2,2)
    # helps gradient flow if centered around 0
])

def sample_transform(sample):
    '''
    sample format: {"images": BGR np.array([480,640,3]), "actions": np.array([1,2])}
    Transforms image from sample to resnet format
    '''
    rgb_image = cv2.cvtColor(sample["images"], cv2.COLOR_BGR2RGB)
    sample["images"] = transform(rgb_image)
    return sample

def visualize_image(image_tensor):
    '''
    Visualizes a transformed image tensor using matplotlib
    image_tensor: torch.Tensor of shape [C,H,W] normalized with ImageNet stats
    '''
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.clone().detach().cpu()
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    image_np = image.permute(1, 2, 0).numpy()

    plt.imshow(image_np)
    plt.axis('off')
    plt.show() 

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

full_dataset = ZimaTorchDataset(file_path="datasets/data/small.hdf5", 
                                sample_transform=sample_transform,
                                max_cached_episodes=75)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(514)
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

sample_images = next(iter(train_dataloader))["images"].to(device)

visualize_image(torchvision.utils.make_grid(sample_images))

model = ActionResNet().to(device)
mse_loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 4

batch_losses = []
test_batch_losses = []
for epoch in range(num_epochs):
    test_pbar = tqdm(test_dataloader, desc=f"Test {epoch+1}/{num_epochs}")
    epoch_test_losses = []
    for batch in test_pbar:
        images = batch["images"].to(device)
        actions = batch["actions"].to(device)
        with torch.no_grad():
            loss = mse_loss(model(images), actions)
        test_loss = loss.item()
        epoch_test_losses.append(test_loss)
        test_pbar.set_postfix({"loss": f"{test_loss:.4f}"})

    avg_loss = np.mean(epoch_test_losses)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Test Loss: {avg_loss:.4f}")
    test_batch_losses.append(avg_loss)

    batch_start = time.time()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:

        images = batch["images"].to(device)
        actions = batch["actions"].to(device)

        data_time = time.time()

        optimizer.zero_grad()
        loss = mse_loss(model(images), actions)
        loss.backward()
        optimizer.step()
        train_time = time.time()

        batch_losses.append(loss.item())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        batch_start = time.time()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(batch_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_batch_losses, label='Test Loss', color='orange')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
