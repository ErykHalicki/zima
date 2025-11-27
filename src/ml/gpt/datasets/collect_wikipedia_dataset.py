from .wikipedia_scraper.scrape import scrape_wikipedia_topic
from .tokenizer import Tokenizer
from .text_dataset import TextDataset
import numpy as np

# Run using python -m datasets.collect_wikipedia_dataset

# Arguments
# -------------

TOPIC = "GPT-1"
PAGE_COUNT = 100
VOCABULARY_MAX_CORPUS_LENGTH = 10000000
DATASET_PATH = f"~/datasets/data/wikipedia_{TOPIC}.hdf5"

# -------------

wikipedia_page_dictionary = scrape_wikipedia_topic(TOPIC, PAGE_COUNT)
tokenizer = Tokenizer()

sample_corpus = ""

for page in wikipedia_page_dictionary.values():
    if len(sample_corpus) > VOCABULARY_MAX_CORPUS_LENGTH:
        break
    sample_corpus+=page.text

tokenizer.calculate_vocabulary_from_text(sample_corpus)

dataset = TextDataset(DATASET_PATH)
dataset.add_vocabulary(tokenizer.vocabulary_to_numpy())
dataset.add_documents({name: tokenizer.tokenize(page.text) for name, page in wikipedia_page_dictionary.items()})

