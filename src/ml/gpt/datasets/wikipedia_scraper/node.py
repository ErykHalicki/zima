from math import inf
import requests
from bs4 import BeautifulSoup

HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

WIKIPEDIA_PREFIX = 'https://en.wikipedia.org/wiki/'

MAX_LINKS = 10000
# max number of links to explore at each page (chooses first MAX_LINKS links)

class node:
    def __init__(self, link):
        self.neighbors = []
        self.link = link
        self.text = None
        self.page_data_fetched = False

    def valid_link(self, link):
        if (link.startswith('/wiki/')
            and 'Special:' not in link
            and 'Category:' not in link
            and 'Wikipedia:' not in link
            and 'Help:' not in link
            and 'File:' not in link
            and 'Portal:' not in link
            and '(identifier)' not in link
            and 'Main_Page' not in link
            and 'Talk:' not in link
            and '(disambiguation)' not in link
            and 'Template:' not in link
            and link[6:] != self.link):
            return True
        return False

    def get_similarity_to(self, target):
        if not self.page_data_fetched:
            raise Exception(f"Must fetch page data for: {self.link} before trying to get similarity")
        if not target.page_data_fetched:
            raise Exception(f"Must fetch page data for: {target.link} before trying to get similarity")
        node_links, target_links = set([neighbor.link for neighbor in self.neighbors]), set([neighbor.link for neighbor in target.neighbors])
        intersection = node_links.intersection(target_links)
        union = node_links.union(target_links)
        return (len(intersection) / len(union)) if intersection else -inf

    def get_page_data(self):
        self.page_data_fetched = True
        if(len(self.neighbors) != 0 and self.text is not None):
            return False
        try:
            response = requests.get(WIKIPEDIA_PREFIX + self.link, headers=HEADER)
            response.raise_for_status()  # Check for request errors
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract neighbors
            neighbors_set = set()
            neighbor_node_list = []
            for a in soup.find_all('a', href=True):
                if self.valid_link(a['href']) and a['href'][6:] not in neighbors_set:
                    neighbor_node_list.append(node(a['href'][6:]))
                    neighbors_set.add(a['href'][6:])
            self.neighbors = neighbor_node_list[:MAX_LINKS]

            # Extract cleaned page text
            for sup in soup.find_all('sup'):
                sup.decompose()
            paragraphs = soup.find_all('p')
            self.text = '\n'.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")
            self.neighbors = []
            self.text = ""
            return False
