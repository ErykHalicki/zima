from models.gpt import GPT
from datasets.tokenizer import Tokenizer

sample_corpus = """Generative Pre-trained Transformer 1 (GPT-1) was the first of OpenAI's large language models following Google's invention of the transformer architecture in 2017.[2] In June 2018, OpenAI released a paper entitled "Improving Language Understanding by Generative Pre-Training",[3] in which they introduced that initial model along with the general concept of a generative pre-trained transformer.[4]Up to that point, the best-performing neural NLP models primarily employed supervised learning from large amounts of manually labeled data. This reliance on supervised learning limited their use of datasets that were not well-annotated, in addition to making it prohibitively expensive and time-consuming to train"""

tokenizer = Tokenizer()
tokenizer.calculate_vocabulary_from_text(sample_corpus)
print(tokenizer.inverse_vocabulary)

gpt = GPT(6,8,512, tokenizer.vocabulary_length()).to("cuda")
gpt.positional_encodings = gpt.positional_encodings.to("cuda")

for i in range(100):
    batch = tokenizer.tokenize(sample_corpus).unsqueeze(0).to("cuda")
    index = gpt.inference(batch)
    output = tokenizer.inverse_vocabulary[index.cpu().item()]
    sample_corpus+=str(output)

print(sample_corpus)

