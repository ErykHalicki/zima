from models.gpt import GPT
from datasets.tokenizer import Tokenizer
import torch

input = """Generative Pre-trained Transformer 1 (GPT-1) was the first of OpenAI's large language models following Google's invention of the transformer architecture in 2017.[2] In June 2018, OpenAI released a paper entitled "Improving Language Understanding by Generative Pre-Training",[3] in which they introduced that initial model along with the general concept of a generative pre-trained transformer.[4]Up to that point, the best-performing neural NLP models primarily employed supervised learning from large amounts of manually labeled data. This reliance on supervised learning limited their use of datasets that were not well-annotated, in addition to making it prohibitively expensive and time-consuming to train"""

tokenizer = Tokenizer()
tokenizer.calculate_vocabulary_from_text(input)

device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

gpt = GPT(12,12,768, tokenizer.vocabulary_length(),device=device)
print(f"Parameters: {gpt.count_parameters()/1000000.0:.2f} M")

output=""+input
for i in range(10):
    text = tokenizer.tokenize(input).to(device)
    index = gpt.inference(text)
    output = tokenizer.inverse_vocabulary[index]
    output+=str(output)

print(output)
