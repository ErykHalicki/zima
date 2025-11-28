from .models.gpt import GPT
from .datasets.tokenizer import Tokenizer, END_TOKEN_ID
import torch
import os

MODEL_PATH = f"~/model_weights/WALL-E_GPT-1.pt"
CONTEXT_WINDOW = 512
TEMPERATURE = 1.0
MAX_TEXT_LENGTH = 10000

input = "Hey there! Whats your name?"

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

checkpoint = torch.load(os.path.expanduser(MODEL_PATH), map_location=device)
hyperparams = checkpoint['hyperparameters']

tokenizer = Tokenizer()
tokenizer.vocabulary = checkpoint['vocabulary']
tokenizer.inverse_vocabulary = checkpoint['inverse_vocabulary']

gpt = GPT(
    hyperparams['num_layers'],
    hyperparams['num_heads'],
    hyperparams['d_model'],
    hyperparams['vocab_size'],
    device=device
)
gpt.load_state_dict(checkpoint['model_state_dict'])
gpt.eval()

print(f"Loaded model from {MODEL_PATH}")
print(f"Parameters: {gpt.count_parameters()/1000000.0:.2f} M")
print(input, end='',flush=True)
text = torch.from_numpy(tokenizer.tokenize(input, with_end_token=False)).to(device)

while len(text) < MAX_TEXT_LENGTH and END_TOKEN_ID not in text:
    context = text[-CONTEXT_WINDOW:] if len(text) > CONTEXT_WINDOW else text
    index = gpt.inference(context, temperature=TEMPERATURE)
    token_output = str(tokenizer.inverse_vocabulary[index])
    text = torch.cat([text, torch.tensor([index]).to(device)])
    print(token_output,end='',flush=True)

print("\n")
