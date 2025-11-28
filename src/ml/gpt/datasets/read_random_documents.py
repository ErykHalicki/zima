from .tokenizer import Tokenizer
from .text_dataset import TextDataset
import random

# Arguments
# -------------

TOPIC = "GPT-1"
DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_tokenized.hdf5"
NUM_DOCUMENTS = 10
# -------------

tokenizer = Tokenizer()
dataset = TextDataset(DATASET_PATH)

tokenizer.vocabulary_from_numpy(dataset.get_vocabulary())

documents_to_read = random.sample(dataset.get_document_name_list(), k=NUM_DOCUMENTS)
for document_name in documents_to_read:
    print(f"\n\n\n{document_name}:\n")
    print(tokenizer.untokenize(dataset.get_document(document_name)))
