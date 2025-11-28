from .wikipedia_scraper.scrape import scrape_wikipedia_topic
from .tokenizer import Tokenizer
from .text_dataset import TextDataset

# Run using python -m datasets.collect_wikipedia_dataset

# Arguments
# -------------
TOPIC = "WALL-E"
PAGE_COUNT = 25000
MAX_VOCAB_SIZE = 100
VOCABULARY_MAX_CORPUS_LENGTH = 10_000_000
DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}.hdf5"
# -------------

dataset = TextDataset(DATASET_PATH)
tokenizer = Tokenizer()

wikipedia_page_dictionary = scrape_wikipedia_topic(TOPIC, PAGE_COUNT)

sample_corpus = ""

for page in wikipedia_page_dictionary.values():
    if len(sample_corpus) > VOCABULARY_MAX_CORPUS_LENGTH:
        break
    sample_corpus+=page.text

tokenizer.calculate_vocabulary_from_text(sample_corpus[:VOCABULARY_MAX_CORPUS_LENGTH])

dataset.add_vocabulary(tokenizer.vocabulary_to_numpy())
dataset.add_documents({name: tokenizer.tokenize(page.text) for name, page in wikipedia_page_dictionary.items()})

