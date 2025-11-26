from models.gpt import GPT
from datasets.tokenizer import Tokenizer
import torch

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

tokenizer = Tokenizer()
tokenizer.load_vocabulary_from_json("datasets/vocabulary.json")

gpt = GPT(4,4,128, tokenizer.vocabulary_length(), device=device)
print(f"Parameters: {gpt.count_parameters()/1000000.0:.2f} M")

# need a dataset
# need to pre_process the dataset to be tokenized
