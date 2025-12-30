from .tokenizer import Tokenizer
from .text_dataset import TextDataset

# Arguments
# -------------

TOPIC = "GPT-1"
DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_tokenized.hdf5"
DOCUMENT_NAME = "GPT-1"

# -------------

tokenizer = Tokenizer()
dataset = TextDataset(DATASET_PATH)

tokenizer.vocabulary_from_numpy(dataset.get_vocabulary())
print(tokenizer.untokenize(dataset.get_document(DOCUMENT_NAME)))
