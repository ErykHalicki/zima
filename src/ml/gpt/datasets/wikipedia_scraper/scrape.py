from math import inf
from .node import node
from collections import deque
import concurrent.futures
import sys
from tqdm import tqdm
import numpy as np

prefix = 'https://en.wikipedia.org/wiki/'
max_workers = 6

def text_to_unicode_array(text):
    return np.array([ord(c) for c in text], dtype=np.int32)

def scrape_wikipedia_topic(start_link: str, max_pages = 100, show_progress_bar = True, dataset = None):
    '''
    start_link: string corresponding to wikipedia page title (not including wikipedia link)
    ex. start_link = "GPT-1" corresponds to the page https://en.wikipedia.org/wiki/GPT-1

    Returns: a dictionary of node objects {node.link: node}. Access text using node.text

    Algorithm:
        Start at start_link, and explore every link on the page
        Then select the link that is most similar to the start page (based on Jaccard Similarity of links)
        Repeat and save data, exploring pages similar to the starting one
    '''
    start = node(start_link)
    current_page = start
    visited_page_dict = {}
    total_word_count = 0
    saved_documents = set()

    def format_number(num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)

    if show_progress_bar:
        pbar = tqdm(total=max_pages, desc="Scraping pages", unit="pages")
    else:
        pbar = None

    def fetch_neighbor_batch(nodes):
        nonlocal total_word_count
        unvisited = deque(nodes)
        while len(unvisited) > 0:
            def pop_and_fetch_neighbors():
                nonlocal total_word_count
                neighbor = unvisited.popleft()
                if len(visited_page_dict) >= max_pages:
                    return
                if neighbor.get_page_data():
                    visited_page_dict[neighbor.link] = neighbor
                    total_word_count += len(neighbor.text.split())
                    if pbar:
                        pbar.n = len(visited_page_dict)
                        pbar.set_postfix_str(f"{format_number(total_word_count)} words")
                        pbar.refresh()
                return

            pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            for _ in range(min(len(unvisited), max_workers)):
                pool.submit(pop_and_fetch_neighbors)
            pool.shutdown(wait=True)

            if dataset is not None:
                for page_link, page_node in visited_page_dict.items():
                    if page_link in saved_documents:
                        continue
                    unicode_array = text_to_unicode_array(page_node.text)
                    dataset.add_document(page_link, unicode_array)
                    saved_documents.add(page_link)

    start.get_page_data()
    visited_page_dict[start.link] = start
    total_word_count += len(start.text.split())
    if pbar:
        pbar.n = len(visited_page_dict)
        pbar.set_postfix_str(f"{format_number(total_word_count)} words")
        pbar.refresh()

    if dataset is not None:
        unicode_array = text_to_unicode_array(start.text)
        dataset.add_document(start.link, unicode_array)
        saved_documents.add(start.link)

    while len(visited_page_dict) < max_pages:
        already_fetched_neighbors = {}
        for index, neighbor in enumerate(current_page.neighbors):
            if neighbor.link in visited_page_dict:
                already_fetched_neighbors[neighbor.link] = index
        for neighbor_link in already_fetched_neighbors:
            current_page.neighbors[already_fetched_neighbors[neighbor_link]] = visited_page_dict[neighbor_link]
            # prevents duplicating node objects
        fetch_neighbor_batch(current_page.neighbors[:max_pages - len(visited_page_dict)])
        best_neighbor = current_page.neighbors[0]
        best_similarity = -inf
        for neighbor in current_page.neighbors:
            if len(visited_page_dict) >= max_pages:
                break
            if neighbor.link in visited_page_dict:
                start_similarity = neighbor.get_similarity_to(start)
                if start_similarity > best_similarity and neighbor.link not in already_fetched_neighbors:
                    best_neighbor = neighbor
        current_page = best_neighbor

    if pbar:
        pbar.close()

    return visited_page_dict

if __name__ == "__main__":
    if len(sys.argv) == 3:
        scrape_wikipedia_topic(sys.argv[1], int(sys.argv[2]))
    else:
        print("Usage: scape.py LINK_TITLE MAX_PAGES")

