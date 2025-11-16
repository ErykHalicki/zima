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

    # bin all actions (action chunk and action history) into 1 of 5 classes (forward left right backward stop)
    # action_history shape: [history_size, 2] -> [history_size, 5] (one-hot)
    # action_chunk shape: [chunk_size, 2] -> [chunk_size, 1] (class index)
    sample["action_history"] = np.array([ActionResNet.bin_action(action) for action in sample["action_history"]], dtype=np.float32)
    sample["action_chunk"] = np.array([np.argmax(ActionResNet.bin_action(action)) for action in sample["action_chunk"]], dtype=np.int64)

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

ACTION_CHUNK_SIZE = 4
ACTION_HISTORY_SIZE = 4
ACTION_SIZE = 4

full_dataset = ZimaTorchDataset(file_path="datasets/data/final!.hdf5", 
                                sample_transform=sample_transform,
                                max_cached_episodes=150,
                                max_cached_images = 0,
                                action_chunk_size = ACTION_CHUNK_SIZE,
                                action_history_size = ACTION_HISTORY_SIZE)

train_size = int(0.70 * len(full_dataset))
test_size = int(0.3 * len(full_dataset))
null_set_size = len(full_dataset) - test_size - train_size

train_dataset, test_dataset, null_set = random_split(
    full_dataset, 
    [train_size, test_size, null_set_size],
    generator=torch.Generator().manual_seed(514)
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory = True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

sample_images = next(iter(train_dataloader))["image"].to(device)

visualize_image(torchvision.utils.make_grid(sample_images))

all_actions = []
for i in range(full_dataset.num_episodes):
    episode_actions = full_dataset.read_specific_key(i,"actions")
    binned_actions = np.array([np.argmax(ActionResNet.bin_action(action)) for action in episode_actions], dtype=np.int64)
    all_actions.append(binned_actions)

all_actions = np.concatenate(all_actions, axis=0)
unique, counts = np.unique(all_actions, axis=0, return_counts=True)
class_weights = []
for action, count in zip(unique, counts):
    print(f"action {action}: {count} ({100*count/len(all_actions):.1f}%) of data")
    class_weights.append((len(all_actions)/count)**(1/2))

class_weights = torch.from_numpy(np.array(class_weights, dtype=np.float32)).to(device)
print(f"class weights: {class_weights}")

model = ActionResNet(ACTION_CHUNK_SIZE, ACTION_HISTORY_SIZE, ACTION_SIZE).to(device)
#loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
loss_criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW([
    {'params': model.feature_extractor.parameters(), 'lr': 1e-5},      # pretrained, small updates
    {'params': model.action_head.parameters(), 'lr': 3e-4}   # random init, larger updates
], weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

num_epochs = 10
max_plateaued_epochs = 5
patience_counter = 0

batch_losses = []
epoch_test_losses = []
prediction_variances = []
ground_truth_variances = []
best_test_loss = 1000.0

MODEL_SAVE_PATH = "models/weights/"
MODEL_NAME = "action_resnet"
CLASS_NAMES = ["stop", "forward", "right", "left"]

for epoch in range(num_epochs):
    batch_start = time.time()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    model.train()
    training_epoch_loss = []
    train_all_predictions = []
    train_all_targets = []
    for batch in pbar:
        images = batch["image"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)

        #action_variance = torch.var(action_chunks).cpu()
        #ground_truth_variances.append(action_variance)

        data_time = time.time()
        
        optimizer.zero_grad()
        predictions = model(images, action_histories)

        #prediction_variance = torch.var(predictions.detach()).cpu()
        #prediction_variances.append(prediction_variance)
        
        logits_flat = predictions.view(-1, ACTION_SIZE)
        targets_flat = action_chunks.view(-1)
        loss = loss_criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        train_time = time.time()

        # Track predictions and targets for accuracy calculation
        predicted_classes = torch.argmax(logits_flat, dim=1)
        train_all_predictions.append(predicted_classes.cpu())
        train_all_targets.append(targets_flat.cpu())

        batch_losses.append(loss.item())
        training_epoch_loss.append(loss.item())
        
        postfix_dict = {"loss": f'{np.mean(training_epoch_loss):.4f}', 
                        "data_time": f"{(data_time-batch_start):.2f}", 
                        "train_time": f"{(train_time-data_time):.2f}"}
        pbar.set_postfix(postfix_dict)
        batch_start = time.time()

    # Calculate and print training per-class accuracy
    train_all_predictions = torch.cat(train_all_predictions)
    train_all_targets = torch.cat(train_all_targets)

    print(f"\nEpoch {epoch+1}/{num_epochs} - Training Metrics:")
    print(f"  Overall Accuracy: {(train_all_predictions == train_all_targets).float().mean():.4f}")
    print("  Per-class Accuracy:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = train_all_targets == class_idx
        if class_mask.sum() > 0:
            class_acc = (train_all_predictions[class_mask] == train_all_targets[class_mask]).float().mean()
            class_count = class_mask.sum().item()
            print(f"    {class_name:8s} ({class_idx}): {class_acc:.4f} ({class_count:5d} samples)")
        else:
            print(f"    {class_name:8s} ({class_idx}): N/A (0 samples)")

    test_pbar = tqdm(test_dataloader, desc=f"Test {epoch+1}/{num_epochs}")
    batch_test_losses = []
    test_all_predictions = []
    test_all_targets = []
    model.eval()
    for batch in test_pbar:
        images = batch["image"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)
        
        with torch.no_grad():
            predictions = model(images, action_histories)
            logits_flat = predictions.view(-1, ACTION_SIZE)
            targets_flat = action_chunks.view(-1)
            loss = loss_criterion(logits_flat, targets_flat)

            # Track predictions and targets for accuracy calculation
            predicted_classes = torch.argmax(logits_flat, dim=1)
            test_all_predictions.append(predicted_classes.cpu())
            test_all_targets.append(targets_flat.cpu())

        test_loss = loss.item()
        batch_test_losses.append(test_loss)
        test_pbar.set_postfix({"loss": f"{np.mean(batch_test_losses):.4f}"})

    # Calculate and print test per-class accuracy
    test_all_predictions = torch.cat(test_all_predictions)
    test_all_targets = torch.cat(test_all_targets)

    avg_test_loss = np.mean(batch_test_losses)
    print(f"\nEpoch {epoch+1}/{num_epochs} - Test Metrics:")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")
    print(f"  Overall Accuracy: {(test_all_predictions == test_all_targets).float().mean():.4f}")
    print("  Per-class Accuracy:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = test_all_targets == class_idx
        if class_mask.sum() > 0:
            class_acc = (test_all_predictions[class_mask] == test_all_targets[class_mask]).float().mean()
            class_count = class_mask.sum().item()
            print(f"    {class_name:8s} ({class_idx}): {class_acc:.4f} ({class_count:5d} samples)")
        else:
            print(f"    {class_name:8s} ({class_idx}): N/A (0 samples)")

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

plt.tight_layout()
plt.show()
