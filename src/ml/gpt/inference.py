from models.gpt import GPT
from datasets.tokenizer import Tokenizer
import torch

TOPIC = "GPT-1"
MODEL_PATH = f"models/weights/{TOPIC}_GPT.pt"

input = """released a paper entitled "Improving La"""

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

checkpoint = torch.load(MODEL_PATH, map_location=device)
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
text = torch.from_numpy(tokenizer.tokenize(input))[:-1].to(device)

for i in range(100):
    context = text[-128:] if len(text) > 128 else text
    index = gpt.inference(context, temperature=0.01)
    token_output = str(tokenizer.inverse_vocabulary[index])
    text = torch.cat([text, torch.tensor([index]).to(device)])
    print(token_output,end='',flush=True)

print("\n")
