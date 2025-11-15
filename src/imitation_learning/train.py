import torch
import torchvision
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

def sample_transform(sample):
    '''
    sample format: {"images": BGR np.array([480,640,3]), "actions": np.array([1,2])}
    Transforms image from sample to resnet format
    '''
    rgb_image = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB)
    sample["image"] = ActionResNet.convert_image_to_resnet(rgb_image)
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

ACTION_CHUNK_SIZE = 8
ACTION_HISTORY_SIZE = 4
ACTION_SIZE = 2

full_dataset = ZimaTorchDataset(file_path="datasets/data/compressed_clockwise.hdf5", 
                                sample_transform=sample_transform,
                                max_cached_episodes=10,
                                max_cached_images = 20000,
                                action_chunk_size = ACTION_CHUNK_SIZE,
                                action_history_size = ACTION_HISTORY_SIZE)

train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(420)
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

sample_images = next(iter(train_dataloader))["image"].to(device)

visualize_image(torchvision.utils.make_grid(sample_images))


model = ActionResNet(ACTION_CHUNK_SIZE, ACTION_HISTORY_SIZE, ACTION_SIZE).to(device)
mse_loss = nn.MSELoss()

optimizer = optim.AdamW([
    {'params': model.feature_extractor.parameters(), 'lr': 1e-4},      # pretrained, small updates
    {'params': model.action_head.parameters(), 'lr': 1e-3}   # random init, larger updates
], weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

num_epochs = 100
max_plateaued_epochs = 15
patience_counter = 0

batch_losses = []
epoch_test_losses = []
prediction_variances = []
ground_truth_variances = []
best_test_loss = 1000.0

MODEL_SAVE_PATH = "models/weights/"
MODEL_NAME = "action_resnet"
for epoch in range(num_epochs):
    batch_start = time.time()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()
    for batch in pbar:

        images = batch["image"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)

        action_variance = torch.var(action_chunks).cpu()
        ground_truth_variances.append(action_variance)

        data_time = time.time()
        
        optimizer.zero_grad()
        predictions = model(images, action_histories)

        prediction_variance = torch.var(predictions.detach()).cpu()
        prediction_variances.append(prediction_variance)

        loss = mse_loss(predictions, action_chunks)
        loss.backward()
        optimizer.step()
        train_time = time.time()

        batch_losses.append(loss.item())
        
        postfix_dict = {"loss": f'{loss.item():.4f}', 
                        "data_time": f"{(data_time-batch_start):.2f}", 
                        "train_time": f"{(train_time-data_time):.2f}"}
        pbar.set_postfix(postfix_dict)
        batch_start = time.time()

    test_pbar = tqdm(test_dataloader, desc=f"Test {epoch+1}/{num_epochs}")
    batch_test_losses = []
    model.eval()
    for batch in test_pbar:
        images = batch["image"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)
        
        with torch.no_grad():
            loss = mse_loss(model(images, action_histories), action_chunks)
        test_loss = loss.item()
        batch_test_losses.append(test_loss)
        test_pbar.set_postfix({"loss": f"{test_loss:.4f}"})

    avg_test_loss = np.mean(batch_test_losses)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Test Loss: {avg_test_loss:.4f}")
    epoch_test_losses.append(avg_test_loss)

    if avg_test_loss <= best_test_loss:
        torch.save(model.state_dict(), MODEL_SAVE_PATH+MODEL_NAME+"_best.pt")
        best_test_loss = avg_test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > max_plateaued_epochs:
            break

    scheduler.step(avg_test_loss)

    torch.save(model.state_dict(), MODEL_SAVE_PATH+MODEL_NAME+"_latest.pt")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(batch_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epoch_test_losses, label='Test Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(prediction_variances, label='Prediction Variance', color='green', alpha=0.7)
plt.plot(ground_truth_variances, label='Ground Truth Variance', color='red', alpha=0.7)
plt.xlabel('Batch')
plt.ylabel('Variance')
plt.title('Prediction vs Ground Truth Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
