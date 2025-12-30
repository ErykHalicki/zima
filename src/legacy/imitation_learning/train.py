import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use('Agg')  
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
from datetime import datetime
import os

def sample_transform(sample):
    transformed_images = [ActionResNet.convert_image_to_resnet(img) for img in sample["images"]]
    sample["images"] = torch.stack(transformed_images)

    sample["action_history"] = np.array([ActionResNet.bin_action(action) for action in sample["action_history"]], dtype=np.float32)
    sample["action_chunk"] = np.array([np.argmax(ActionResNet.bin_action(action)) for action in sample["action_chunk"]], dtype=np.int64)

    return sample

def visualize_image(image_tensor, save_path):
    '''
    Visualizes a transformed image tensor using matplotlib and saves to file
    image_tensor: torch.Tensor of shape [C,H,W] normalized with ImageNet stats
    save_path: path to save the plot
    '''
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.clone().detach().cpu()
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    image_np = image.permute(1, 2, 0).numpy()

    plt.figure()
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved sample images to {save_path}") 

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

#RESUME_MODEL_PATH = None
RESUME_MODEL_PATH = "models/weights/action_resnet_latest.pt"
ACTION_CHUNK_SIZE = 10
ACTION_HISTORY_SIZE = 20
ACTION_SIZE = 4
IMAGE_HISTORY_SIZE = 3
DATASET_PATH = "datasets/data/rubiks_cube_navigation_real_resized.hdf5"

if RESUME_MODEL_PATH is not None:
    print(f"\nLoading model configuration from: {RESUME_MODEL_PATH}")
    checkpoint = torch.load(RESUME_MODEL_PATH, weights_only=False, map_location=device)
    metadata = checkpoint.get('metadata', {})
    ACTION_CHUNK_SIZE = metadata.get('action_chunk_size', ACTION_CHUNK_SIZE)
    ACTION_HISTORY_SIZE = metadata.get('action_history_size', ACTION_HISTORY_SIZE)
    ACTION_SIZE = metadata.get('action_size', ACTION_SIZE)
    IMAGE_HISTORY_SIZE = metadata.get('image_history_size', IMAGE_HISTORY_SIZE)
    print(f"Loaded hyperparameters from checkpoint:")
else:
    print(f"Loaded default hyperparameters:")

print(f"\tACTION_CHUNK_SIZE: {ACTION_CHUNK_SIZE}")
print(f"\tACTION_HISTORY_SIZE: {ACTION_HISTORY_SIZE}")
print(f"\tACTION_SIZE: {ACTION_SIZE}")
print(f"\tIMAGE_HISTORY_SIZE: {IMAGE_HISTORY_SIZE}")

print(f"Using dataset: {DATASET_PATH}")

full_dataset = ZimaTorchDataset(file_path=DATASET_PATH,
                                sample_transform=sample_transform,
                                max_cached_episodes=300,
                                max_cached_images = 0,
                                action_chunk_size = ACTION_CHUNK_SIZE,
                                action_history_size = ACTION_HISTORY_SIZE,
                                image_history_size = IMAGE_HISTORY_SIZE)

train_size = int(0.8 * len(full_dataset))
test_size = int(0.2 * len(full_dataset))
null_set_size = len(full_dataset) - test_size - train_size

train_dataset, test_dataset, null_set = random_split(
    full_dataset,
    [train_size, test_size, null_set_size],
    generator=torch.Generator().manual_seed(514)
)

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True, pin_memory = True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True, pin_memory = True)

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

model = ActionResNet(ACTION_CHUNK_SIZE, ACTION_HISTORY_SIZE, ACTION_SIZE, IMAGE_HISTORY_SIZE).to(device)

# Load existing weights if RESUME_MODEL_PATH is set
if RESUME_MODEL_PATH is not None:
    print(f"\nLoading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded weights from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_test_loss' in checkpoint:
        print(f"Previous best test loss: {checkpoint['best_test_loss']:.4f}")
    print("Resuming training from checkpoint\n")

loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
#loss_criterion = nn.CrossEntropyLoss()

if RESUME_MODEL_PATH == None:
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-5},      # pretrained, small updates
        {'params': model.action_head.parameters(), 'lr': 1e-4}   # random init, larger updates
    ], weight_decay=0.01)
else:
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-7},      # pretrained, small updates
        {'params': model.action_head.parameters(), 'lr': 1e-6}   # random init, larger updates
    ], weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

num_epochs = 15
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

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
PLOTS_DIR = os.path.join(MODEL_SAVE_PATH, "plots",TIMESTAMP)
os.makedirs(PLOTS_DIR, exist_ok=True)

sample_images = next(iter(train_dataloader))["images"].to(device)
sample_images_latest = sample_images[:, -1, :, :, :]

sample_images_path = os.path.join(PLOTS_DIR, f"sample_images.png")
visualize_image(torchvision.utils.make_grid(sample_images_latest), sample_images_path)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model_metadata = {
    'action_chunk_size': ACTION_CHUNK_SIZE,
    'action_history_size': ACTION_HISTORY_SIZE,
    'action_size': ACTION_SIZE,
    'image_history_size': IMAGE_HISTORY_SIZE,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'total_dataset_size': len(full_dataset),
    'train_dataset_size': train_size,
    'test_dataset_size': test_size,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_architecture': 'ActionResNet',
    'class_names': CLASS_NAMES
}

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Dataset size: {len(full_dataset):,} samples")
print(f"  Training started: {model_metadata['training_date']}")

for epoch in range(num_epochs):
    batch_start = time.time()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=0.5)

    model.train()
    training_epoch_loss = []
    train_all_predictions = []
    train_all_targets = []
    for batch in pbar:
        images = batch["images"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)

        data_time = time.time()
        
        optimizer.zero_grad()
        predictions = model(images, action_histories)

        logits_flat = predictions.view(-1, ACTION_SIZE)
        targets_flat = action_chunks.view(-1)
        loss = loss_criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        train_time = time.time()

        predicted_classes = torch.argmax(logits_flat, dim=1)
        train_all_predictions.append(predicted_classes)
        train_all_targets.append(targets_flat)

        training_epoch_loss.append(loss.detach())

        overhead_start = time.time()
        postfix_dict = {"data_time": f"{(data_time-batch_start):.2f}",
                        "train_time": f"{(train_time-data_time):.2f}",
                        "overhead": f"{(overhead_start-train_time):.2f}"}
        pbar.set_postfix(postfix_dict)
        batch_start = time.time()

    train_all_predictions = torch.cat(train_all_predictions).cpu()
    train_all_targets = torch.cat(train_all_targets).cpu()
    training_epoch_loss_cpu = [l.item() for l in training_epoch_loss]
    batch_losses.extend(training_epoch_loss_cpu)

    print(f"\nEpoch {epoch+1}/{num_epochs} - Training Metrics:")
    print(f"  Average Training Loss: {np.mean(training_epoch_loss_cpu):.4f}")
    print(f"  Overall TRAIN Accuracy: {(train_all_predictions == train_all_targets).float().mean():.4f}")
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
        images = batch["images"].to(device)
        action_histories = batch["action_history"].to(device)
        action_chunks = batch["action_chunk"].to(device)
        
        with torch.no_grad():
            predictions = model(images, action_histories)
            logits_flat = predictions.view(-1, ACTION_SIZE)
            targets_flat = action_chunks.view(-1)
            loss = loss_criterion(logits_flat, targets_flat)

            predicted_classes = torch.argmax(logits_flat, dim=1)
            test_all_predictions.append(predicted_classes)
            test_all_targets.append(targets_flat)

        test_loss = loss.item()
        batch_test_losses.append(test_loss)
        epoch_test_losses.append(test_loss)
        test_pbar.set_postfix({"loss": f"{np.mean(batch_test_losses):.4f}"})

    test_all_predictions = torch.cat(test_all_predictions).cpu()
    test_all_targets = torch.cat(test_all_targets).cpu()

    avg_test_loss = np.mean(batch_test_losses)
    print(f"\nEpoch {epoch+1}/{num_epochs} - Test Metrics:")
    print(f"  Overall TEST Accuracy: {(test_all_predictions == test_all_targets).float().mean():.4f}")
    print("  Per-class Accuracy:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_mask = test_all_targets == class_idx
        if class_mask.sum() > 0:
            class_acc = (test_all_predictions[class_mask] == test_all_targets[class_mask]).float().mean()
            class_count = class_mask.sum().item()
            print(f"    {class_name:8s} ({class_idx}): {class_acc:.4f} ({class_count:5d} samples)")
        else:
            print(f"    {class_name:8s} ({class_idx}): N/A (0 samples)")


    if avg_test_loss <= best_test_loss:
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': model_metadata,
            'epoch': epoch + 1,
            'best_test_loss': avg_test_loss
        }, MODEL_SAVE_PATH+MODEL_NAME+"_best.pt")
        best_test_loss = avg_test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > max_plateaued_epochs:
            break

    scheduler.step(avg_test_loss)

    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': model_metadata,
        'epoch': epoch + 1,
        'last_test_loss': avg_test_loss
    }, MODEL_SAVE_PATH+MODEL_NAME+"_latest.pt")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 5, 1)
    plt.plot(batch_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 5, 2)
    plt.plot(epoch_test_losses, label='Test Loss', color='orange')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    training_loss_plot_path = os.path.join(PLOTS_DIR, f"training_loss.png")
    plt.savefig(training_loss_plot_path, dpi=150)
    plt.close()
    print(f"Saved training loss plots to {training_loss_plot_path}")



