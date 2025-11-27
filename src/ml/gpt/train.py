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

TOPIC = "GPT-1"
DATASET_PATH = f"~/datasets/data/wikipedia_{TOPIC}.hdf5"
MODEL_PATH = "models/weights/"
LOAD_MODEL = None
#LOAD_MODEL = "models/weights/model_epoch_2.pt"
CHUNK_SIZE = 128
EPOCHS = 300
NUM_LAYERS = 8
NUM_HEADS = 4
D_MODEL = 256
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 1
WARMUP_EPOCHS = 30

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

full_dataset = TorchTextDataset(DATASET_PATH, chunk_size=CHUNK_SIZE)
tokenizer = Tokenizer()
tokenizer.vocabulary_from_numpy(full_dataset.get_vocabulary())

start_epoch = 0
if LOAD_MODEL:
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

'''
train_size = int( * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(514)
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True)
'''
dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory = True, num_workers=4, persistent_workers=True)

loss_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler(device.type)

warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=0)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS])

if LOAD_MODEL:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    del checkpoint

print(tokenizer.untokenize(full_dataset[0]['chunks'].numpy()))

model.train()
for epoch in range(EPOCHS):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        with torch.amp.autocast(device.type):
            batch_chunks = batch['chunks'].to(device)
            batch_padding_masks = batch['masks'].to(device)
            #print(batch_padding_masks[0])

            seq_len = batch_chunks.shape[1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)
            #lower triangular matrix forcing model to only attend to current and past tokens

            padding_mask = batch_padding_masks.unsqueeze(1) # (batch, d_seq) -> (batch, 1, d_seq)
            combined_mask = causal_mask * padding_mask # removes 1's from causal mask at token positions that are masked

            #print(combined_mask)

            optimizer.zero_grad()
            logits = model(batch_chunks, mask=combined_mask)

            # for token sequence ABCD
            target_tokens = batch_chunks[:, 1:]# ABCD -> BCD
            logits = logits[:, :-1, :] # BCDE -> BCD

            batch_size, seq_len_minus_1, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len_minus_1, vocab_size)
            targets_flat = target_tokens.reshape(batch_size * seq_len_minus_1)

            loss = loss_criterion(logits_flat, targets_flat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

    scheduler.step()

    if epoch % 30 == 0:
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
        torch.save(checkpoint, os.path.join(MODEL_PATH, f'{TOPIC}_GPT.pt'))

