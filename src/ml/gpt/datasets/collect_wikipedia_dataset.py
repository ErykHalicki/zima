from .wikipedia_scraper.scrape import scrape_wikipedia_topic
from .text_dataset import TextDataset
import os

# Run using python -m datasets.collect_wikipedia_dataset

# Arguments
# -------------
TOPIC = "WALL-E"
PAGE_COUNT = 100000
DATASET_PATH = f"~/datasets/wikipedia_{TOPIC}_unicode.hdf5"
STATE_PATH = f"~/datasets/wikipedia_{TOPIC}_scrape_state.json"
# -------------

dataset = TextDataset(DATASET_PATH, unicode_vocabulary=True)

wikipedia_page_dictionary = scrape_wikipedia_topic(TOPIC, PAGE_COUNT, dataset=dataset, state_file=os.path.expanduser(STATE_PATH))
