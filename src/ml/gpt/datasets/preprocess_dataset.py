from .text_dataset import TextDataset
from .tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm

INPUT_DATASET_PATH = "~/datasets/wikipedia_WALL-E_unicode.hdf5"
OUTPUT_DATASET_PATH = "~/datasets/wikipedia_WALL-E_tokenized.hdf5"
MAX_VOCAB_SIZE = 150

input_dataset = TextDataset(INPUT_DATASET_PATH)

if not input_dataset.unicode_vocabulary:
    raise Exception("Input dataset must be in unicode_vocabulary mode. This script only processes unicode datasets.")

print(f"Loading unicode dataset from: {input_dataset.file_path}")

document_names = input_dataset.get_document_name_list()
print(f"Found {len(document_names)} documents")

print("Reading all documents and building vocabulary...")
all_text = ""
for doc_name in tqdm(document_names, desc="Reading documents", unit="docs"):
    unicode_array = input_dataset.get_document(doc_name)
    text = ''.join([chr(code) for code in unicode_array])
    all_text += text

print(f"Total text length: {len(all_text)} characters")

tokenizer = Tokenizer()
tokenizer.calculate_vocabulary_from_text(all_text, max_vocabulary_size=MAX_VOCAB_SIZE)
print(f"Vocabulary size: {tokenizer.vocabulary_length()}")

output_dataset = TextDataset(OUTPUT_DATASET_PATH, unicode_vocabulary=False)
output_dataset.add_vocabulary(tokenizer.vocabulary_to_numpy())
print(f"Created output dataset with vocabulary at: {output_dataset.file_path}")

print(''.join(tokenizer.vocabulary.keys()))

print("Tokenizing and writing documents to new dataset...")
for doc_name in tqdm(document_names, desc="Tokenizing documents", unit="docs"):
    unicode_array = input_dataset.get_document(doc_name)
    text = ''.join([chr(code) for code in unicode_array])
    tokenized = tokenizer.tokenize(text)
    output_dataset.add_document(doc_name, tokenized)

print(f"Preprocessing complete! Output saved to: {output_dataset.file_path}")
