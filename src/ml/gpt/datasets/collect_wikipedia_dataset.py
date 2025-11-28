from .wikipedia_scraper.scrape import scrape_wikipedia_topic
from .text_dataset import TextDataset

# Run using python -m datasets.collect_wikipedia_dataset

# Arguments
# -------------
TOPIC = "WALL-E"
PAGE_COUNT = 25000
DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_unicode.hdf5"
# -------------

dataset = TextDataset(DATASET_PATH, unicode_vocabulary=True)

wikipedia_page_dictionary = scrape_wikipedia_topic(TOPIC, PAGE_COUNT, dataset=dataset)
