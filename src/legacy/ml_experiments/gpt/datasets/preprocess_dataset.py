from .text_dataset import TextDataset
from .tokenizer import Tokenizer
from tqdm import tqdm

TOPIC = "WALL-E"
INPUT_DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_unicode.hdf5"
OUTPUT_DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_tokenized.hdf5"
JSON_PATH = f"~/datasets/wikipedia_{TOPIC}_token_frequencies.json"
FROM_JSON = True
MAX_VOCAB_SIZE = 120

tokenizer = Tokenizer()
input_dataset = TextDataset(INPUT_DATASET_PATH)

if not FROM_JSON:
    tokenizer.calculate_vocabulary_from_unicode_dataset(input_dataset, max_vocabulary_size=MAX_VOCAB_SIZE, json_save_path=JSON_PATH)
else:
    tokenizer.calculate_vocabulary_from_json(JSON_PATH, max_vocabulary_size= MAX_VOCAB_SIZE)
print(f"Vocabulary size: {tokenizer.vocabulary_length()}")

output_dataset = TextDataset(OUTPUT_DATASET_PATH, unicode_vocabulary=False)
output_dataset.add_vocabulary(tokenizer.vocabulary_to_numpy())
print(f"Created output dataset with vocabulary at: {output_dataset.file_path}")

print(''.join(tokenizer.vocabulary.keys()))

document_names = input_dataset.get_document_name_list()

print("Tokenizing and writing documents to new dataset...")
for doc_name in tqdm(document_names, desc="Tokenizing documents", unit="docs"):
    unicode_array = input_dataset.get_document(doc_name)
    text = ''.join([chr(code) for code in unicode_array])
    tokenized = tokenizer.tokenize(text)
    output_dataset.add_document(doc_name, tokenized)

print(f"Preprocessing complete! Output saved to: {output_dataset.file_path}")
