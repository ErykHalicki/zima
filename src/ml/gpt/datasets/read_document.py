from .tokenizer import Tokenizer
from .text_dataset import TextDataset
import numpy as np

# Arguments
# -------------

TOPIC = "GPT-1"
DATASET_PATH = f"~/datasets/data/wikipedia_{TOPIC}.hdf5"
DOCUMENT_NAME = "GPT-1"

# -------------

tokenizer = Tokenizer()
dataset = TextDataset(DATASET_PATH)

tokenizer.vocabulary_from_numpy(dataset.get_vocabulary())
print(tokenizer.untokenize(dataset.get_document(DOCUMENT_NAME)))
