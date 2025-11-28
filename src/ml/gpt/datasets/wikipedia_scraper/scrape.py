from math import inf
from .node import node
from collections import deque
import concurrent.futures
import sys
from tqdm import tqdm
import numpy as np
import json
import os
import heapq

prefix = 'https://en.wikipedia.org/wiki/'
MAX_WORKERS = 4
SAVE_INTERVAL = 1500

def text_to_unicode_array(text):
    return np.array([ord(c) for c in text], dtype=np.int32)

def save_state(state_file, visited_page_dict, current_page_link, start_link, saved_documents, total_word_count, clicked, similarity_scores):
    state = {
        'start_link': start_link,
        'current_page_link': current_page_link,
        'total_word_count': total_word_count,
        'saved_documents': list(saved_documents),
        'clicked': list(clicked),
        'similarity_scores': similarity_scores,
        'visited_pages': {
            link: node_obj.neighbors
            for link, node_obj in visited_page_dict.items()
        }
    }
    with open(state_file, 'w') as f:
        json.dump(state, f)

def load_state(state_file):
    if not os.path.exists(state_file):
        return None
    with open(state_file, 'r') as f:
        state = json.load(f)

    visited_page_dict = {}
    for link, neighbor_links in state['visited_pages'].items():
        n = node(link)
        n.neighbors = neighbor_links
        n.page_data_fetched = True
        visited_page_dict[link] = n

    return {
        'visited_page_dict': visited_page_dict,
        'current_page_link': state['current_page_link'],
        'start_link': state['start_link'],
        'saved_documents': set(state['saved_documents']),
        'total_word_count': state['total_word_count'],
        'clicked': set(state.get('clicked', [])),
        'similarity_scores': state.get('similarity_scores', {})
    }

def scrape_wikipedia_topic(start_link: str, max_pages = 100, show_progress_bar = True, dataset = None, state_file = None):
    '''
    start_link: string corresponding to wikipedia page title (not including wikipedia link)
    ex. start_link = "GPT-1" corresponds to the page https://en.wikipedia.org/wiki/GPT-1

    Returns: a dictionary of node objects {node.link: node}. Access text using node.text

    Algorithm:
        Start at start_link, and explore every link on the page
        Then select the link that is most similar to the start page (based on Jaccard Similarity of links)
        Repeat and save data, exploring pages similar to the starting one
    '''
    loaded_state = load_state(state_file) if state_file else None

    if loaded_state:
        visited_page_dict = loaded_state['visited_page_dict']
        saved_documents = loaded_state['saved_documents']
        total_word_count = loaded_state['total_word_count']
        clicked = loaded_state['clicked']
        similarity_scores = loaded_state['similarity_scores']
        start = visited_page_dict[loaded_state['start_link']]
        current_page = visited_page_dict[loaded_state['current_page_link']]
    else:
        start = node(start_link)
        current_page = start
        visited_page_dict = {}
        total_word_count = 0
        saved_documents = set()
        clicked = set()
        similarity_scores = {}

    last_saved_page_count = len(visited_page_dict)
    newly_visited_pages = set()
    priority_queue = []

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

        def pop_and_fetch_neighbors():
            nonlocal total_word_count
            try:
                neighbor = unvisited.popleft()
            except IndexError:
                return
            if len(visited_page_dict) >= max_pages:
                return
            if neighbor.get_page_data():
                visited_page_dict[neighbor.link] = neighbor
                newly_visited_pages.add(neighbor.link)
                total_word_count += len(neighbor.text.split())
                if pbar:
                    pbar.n = len(visited_page_dict)
                    pbar.set_postfix_str(f"{format_number(total_word_count)} words")
                    pbar.refresh()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(pop_and_fetch_neighbors) for _ in range(len(unvisited))]
            concurrent.futures.wait(futures)

        if dataset is not None:
            for page_link in newly_visited_pages:
                page_node = visited_page_dict[page_link]
                unicode_array = text_to_unicode_array(page_node.text)
                dataset.add_document(page_link, unicode_array)
                saved_documents.add(page_link)
                del page_node.text
            newly_visited_pages.clear()

    if not loaded_state:
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
            del start.text
    else:
        if pbar:
            pbar.n = len(visited_page_dict)
            pbar.set_postfix_str(f"{format_number(total_word_count)} words")
            pbar.refresh()

    clicked.add(current_page.link)

    while len(visited_page_dict) < max_pages:
        neighbor_nodes = []
        for neighbor_link in current_page.neighbors:
            if neighbor_link in visited_page_dict:
                neighbor_nodes.append(visited_page_dict[neighbor_link])
            else:
                neighbor_nodes.append(node(neighbor_link))

        fetch_neighbor_batch(neighbor_nodes[:max_pages - len(visited_page_dict)])

        for neighbor in neighbor_nodes:
            if len(visited_page_dict) >= max_pages:
                break
            if neighbor.link in visited_page_dict and neighbor.link not in similarity_scores and neighbor.link not in clicked:
                similarity = neighbor.get_similarity_to(start)
                similarity_scores[neighbor.link] = similarity
                heapq.heappush(priority_queue, (-similarity, neighbor.link))

        while priority_queue:
            neg_similarity, best_link = heapq.heappop(priority_queue)
            if best_link not in clicked and best_link in visited_page_dict:
                current_page = visited_page_dict[best_link]
                clicked.add(current_page.link)
                print(f"Clicking {current_page.link}")
                break
        else:
            break

        if state_file and (len(visited_page_dict) - last_saved_page_count >= SAVE_INTERVAL or last_saved_page_count == 0):
            save_state(state_file, visited_page_dict, current_page.link, start.link, saved_documents, total_word_count, clicked, similarity_scores)
            last_saved_page_count = len(visited_page_dict)

    if pbar:
        pbar.close()

    if state_file:
        save_state(state_file, visited_page_dict, current_page.link, start.link, saved_documents, total_word_count, clicked, similarity_scores)

    return visited_page_dict

if __name__ == "__main__":
    if len(sys.argv) == 3:
        scrape_wikipedia_topic(sys.argv[1], int(sys.argv[2]))
    else:
        print("Usage: scape.py LINK_TITLE MAX_PAGES")

