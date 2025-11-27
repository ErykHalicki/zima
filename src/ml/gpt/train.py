from models.gpt import GPT
from datasets.tokenizer import Tokenizer, PAD_TOKEN
from datasets.torch_text_dataset import TorchTextDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


DATASET_PATH = "~/datasets/data/wikipedia_Monkey.hdf5"
MODEL_PATH = "models/weights/"
LOAD_MODEL = None
CHUNK_SIZE = 256
EPOCHS = 30
NUM_LAYERS = 12
NUM_HEADS = 8
D_MODEL = 512
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 32

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

full_dataset = TorchTextDataset(DATASET_PATH, chunk_size=CHUNK_SIZE)
tokenizer = Tokenizer()
tokenizer.vocabulary_from_numpy(full_dataset.get_vocabulary())

model = GPT(NUM_LAYERS, NUM_HEADS, D_MODEL, tokenizer.vocabulary_length(), device=device)

if LOAD_MODEL:
    checkpoint = torch.load(LOAD_MODEL, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {LOAD_MODEL}")

print(f"Parameters: {model.count_parameters()/1000000.0:.2f} M")

train_size = int(0.999 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(514)
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory = True)


loss_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(EPOCHS):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in progress_bar:
        batch_chunks = batch['chunks'].to(device)
        batch_padding_masks = batch['masks'].to(device)

        seq_len = batch_chunks.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)
        #lower triangular matrix forcing model to only attend to current and past tokens

        padding_mask = batch_padding_masks.unsqueeze(1) # (batch, d_seq) -> (batch, 1, d_seq)
        combined_mask = causal_mask * padding_mask # removes 1's from causal mask at token positions that are masked

        optimizer.zero_grad()

        logits = model(batch_chunks, mask=combined_mask)

        # for token sequence ABCD
        target_tokens = batch_chunks[:, 1:]# ABCD -> BCD
        logits = logits[:, :-1, :] # BCDE -> BCD

        batch_size, seq_len_minus_1, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len_minus_1, vocab_size)
        targets_flat = target_tokens.reshape(batch_size * seq_len_minus_1)

        loss = loss_criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    torch.save(checkpoint, os.path.join(MODEL_PATH, f'model_epoch_{epoch+1}.pt'))

