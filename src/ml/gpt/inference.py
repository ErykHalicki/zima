from models.gpt import GPT
from datasets.tokenizer import Tokenizer
import torch

MODEL_PATH = "models/weights/model_epoch_1.pt"

input = """Hello my name is """

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

for i in range(10):
    text = torch.from_numpy(tokenizer.tokenize(input)).to(device)
    index = gpt.inference(text)
    input+=str(tokenizer.inverse_vocabulary[index])

print(input)
