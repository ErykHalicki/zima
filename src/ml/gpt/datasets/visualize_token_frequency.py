from .text_dataset import TextDataset
from .tokenizer import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

INPUT_DATASET_PATH = "~/datasets/wikipedia_WALL-E_unicode.hdf5"

input_dataset = TextDataset(INPUT_DATASET_PATH)

if not input_dataset.unicode_vocabulary:
    raise Exception("Input dataset must be in unicode_vocabulary mode.")

print(f"Loading unicode dataset from: {input_dataset.file_path}")

document_names = input_dataset.get_document_name_list()
print(f"Found {len(document_names)} documents")

print("Reading all documents and calculating token frequencies...")
tokenizer = Tokenizer()
all_token_freq = {}

for doc_name in tqdm(document_names, desc="Reading documents", unit="docs"):
    unicode_array = input_dataset.get_document(doc_name)
    text = ''.join([chr(code) for code in unicode_array])
    doc_freq = tokenizer.get_token_frequency(text)

    for token, freq in doc_freq.items():
        all_token_freq[token] = all_token_freq.get(token, 0) + freq

sorted_tokens = sorted(all_token_freq.items(), key=lambda x: x[1], reverse=True)

frequencies = [freq for token, freq in sorted_tokens]
total_count = sum(frequencies)

cumulative = np.cumsum(frequencies) / total_count
'''
plt.figure(figsize=(12, 6))
plt.plot(range(len(cumulative)), cumulative)
plt.xlabel('Token Rank (most common to least common)')
plt.ylabel('Cumulative Fraction of Total Tokens')
plt.title('Token Frequency Cumulative Distribution Function')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

print(f"\nTotal unique tokens: {len(sorted_tokens)}")
print(f"Total token count: {total_count}")

print("\nVocabulary size required for coverage:")
coverage_targets = []
for i in range(10, 81, 10):
    coverage_targets.append(i / 100.0)
for i in range(85, 96, 5):
    coverage_targets.append(i / 100.0)
for i in range(96, 99):
    coverage_targets.append(i / 100.0)
for i in range(990, 999):
    coverage_targets.append(i / 1000.0)
for i in range(99900, 100000):
    coverage_targets.append(i / 100000.0)

for target in coverage_targets:
    idx = np.searchsorted(cumulative, target)
    vocab_size = idx + 1
    actual_coverage = cumulative[idx] if idx < len(cumulative) else 1.0
    print(f"  {target*100:5.4f}% coverage: {vocab_size:5d} tokens (actual: {actual_coverage*100:.4f}%)")
