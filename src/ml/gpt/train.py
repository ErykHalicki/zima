from models.gpt import GPT
from datasets.tokenizer import Tokenizer, PAD_TOKEN_ID
from datasets.torch_text_dataset import TorchTextDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import os
import argparse
import yaml
import subprocess
import sys
sys.path.append(".")#hack to use all packages in this directory

try:
    from torch.amp import GradScaler
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
    except ImportError:
        GradScaler = None

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def upload_to_s3(local_path, s3_path):
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', local_path, s3_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully uploaded {local_path} to {s3_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload to S3: {e.stderr}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    MODEL_NAME = config.get('model_name', 'GPT-1')
    DATASET_PATH = os.path.expanduser(config['dataset_path'])
    MODEL_PATH = os.path.expanduser(config['model_path'])
    LOAD_MODEL = os.path.expanduser(config['checkpoint_path']) if config.get('checkpoint_path') else None
    CHUNK_SIZE = config.get('chunk_size', 128)
    EPOCHS = config.get('epochs', 300)
    NUM_LAYERS = config.get('num_layers', 8)
    NUM_HEADS = config.get('num_heads', 4)
    D_MODEL = config.get('d_model', 256)
    LEARNING_RATE = config.get('learning_rate', 2.5e-4)
    BATCH_SIZE = config.get('batch_size', 1)
    WARMUP_EPOCHS = config.get('warmup_epochs', 30)
    NUM_WORKERS = config.get('num_workers', 4)
    CHECKPOINT_INTERVAL = config.get('checkpoint_interval', 30)
    SAVE_TO_S3 = config.get('save_to_s3', False)
    S3_PATH = config.get('s3_path', '')

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    full_dataset = TorchTextDataset(DATASET_PATH, chunk_size=CHUNK_SIZE)
    tokenizer = Tokenizer()
    tokenizer.vocabulary_from_numpy(full_dataset.get_vocabulary())

    val_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    start_epoch = 0
    if LOAD_MODEL and os.path.exists(LOAD_MODEL):
        checkpoint = torch.load(LOAD_MODEL, map_location=device)
        hyperparams = checkpoint['hyperparameters']
        NUM_LAYERS = hyperparams['num_layers']
        NUM_HEADS = hyperparams['num_heads']
        D_MODEL = hyperparams['d_model']
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {LOAD_MODEL}")
        print(f"Hyperparameters: layers={NUM_LAYERS}, heads={NUM_HEADS}, d_model={D_MODEL}")
        print(f"Resuming from epoch {start_epoch}")

    model = GPT(NUM_LAYERS, NUM_HEADS, D_MODEL, tokenizer.vocabulary_length(), device=device)

    if LOAD_MODEL:
        model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Parameters: {model.count_parameters()/1000000.0:.2f} M")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True if NUM_WORKERS > 0 else False)

    loss_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    try:
        scaler = GradScaler(device.type)
    except TypeError:
        scaler = GradScaler()

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

    if LOAD_MODEL:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        del checkpoint

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            with torch.amp.autocast(device.type):
                batch_chunks = batch['chunks'].to(device)
                batch_padding_masks = batch['masks'].to(device)

                seq_len = batch_chunks.shape[1]
                causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)

                padding_mask = batch_padding_masks.unsqueeze(1)
                combined_mask = causal_mask * padding_mask

                logits = model(batch_chunks, mask=combined_mask)

                target_tokens = batch_chunks[:, 1:]
                logits = logits[:, :-1, :]

                batch_size, seq_len_minus_1, vocab_size = logits.shape
                logits_flat = logits.reshape(batch_size * seq_len_minus_1, vocab_size)
                targets_flat = target_tokens.reshape(batch_size * seq_len_minus_1)

                loss = loss_criterion(logits_flat, targets_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                with torch.amp.autocast(device.type):
                    batch_chunks = batch['chunks'].to(device)
                    batch_padding_masks = batch['masks'].to(device)

                    seq_len = batch_chunks.shape[1]
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)

                    padding_mask = batch_padding_masks.unsqueeze(1)
                    combined_mask = causal_mask * padding_mask

                    logits = model(batch_chunks, mask=combined_mask)

                    target_tokens = batch_chunks[:, 1:]
                    logits = logits[:, :-1, :]

                    batch_size, seq_len_minus_1, vocab_size = logits.shape
                    logits_flat = logits.reshape(batch_size * seq_len_minus_1, vocab_size)
                    targets_flat = target_tokens.reshape(batch_size * seq_len_minus_1)

                    loss = loss_criterion(logits_flat, targets_flat)
                    val_loss += loss.item()
                    num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        print(f"Epoch {epoch+1}/{EPOCHS} - Validation Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'hyperparameters': {
                    'num_layers': NUM_LAYERS,
                    'num_heads': NUM_HEADS,
                    'd_model': D_MODEL,
                    'vocab_size': tokenizer.vocabulary_length(),
                    'chunk_size': CHUNK_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'batch_size': BATCH_SIZE
                },
                'vocabulary': tokenizer.vocabulary,
                'inverse_vocabulary': tokenizer.inverse_vocabulary
            }
            local_checkpoint_path = os.path.join(MODEL_PATH, f'{MODEL_NAME}.pt')
            torch.save(checkpoint, local_checkpoint_path)

            if SAVE_TO_S3 and S3_PATH:
                s3_checkpoint_path = os.path.join(S3_PATH, f'{MODEL_NAME}.pt')
                upload_to_s3(local_checkpoint_path, s3_checkpoint_path)

