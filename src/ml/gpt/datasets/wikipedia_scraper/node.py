from math import inf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

HEADER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

WIKIPEDIA_PREFIX = 'https://en.wikipedia.org/wiki/'

MAX_LINKS = 10000

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.1)
adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
session.mount('https://', adapter)
session.headers.update(HEADER)
# max number of links to explore at each page (chooses first MAX_LINKS links)

BANNED_TERMS = [
    'Special:', 'Category:', 'Wikipedia:', 'Help:', 'File:', 'Portal:',
    'Talk:', 'Template:', 'Template_talk:', 'User:', 'User_talk:',
    'MediaWiki:', 'Module:', 'Draft:', 'TimedText:', 'Book:',
    '(identifier)', '(disambiguation)', 'Main_Page'
]

class node:
    def __init__(self, link):
        self.neighbors = []
        self.link = link
        self.text = None
        self.page_data_fetched = False

    def valid_link(self, link):
        if not link.startswith('/wiki/'):
            return False
        if link[6:] == self.link:
            return False
        return not any(term in link for term in BANNED_TERMS)

    def get_similarity_to(self, target):
        if not self.page_data_fetched:
            raise Exception(f"Must fetch page data for: {self.link} before trying to get similarity")
        if not target.page_data_fetched:
            raise Exception(f"Must fetch page data for: {target.link} before trying to get similarity")
        node_links, target_links = set(self.neighbors), set(target.neighbors)
        intersection = node_links.intersection(target_links)
        union = node_links.union(target_links)
        return (len(intersection) / len(union)) if intersection else -inf

    def get_page_data(self):
        self.page_data_fetched = True
        if(len(self.neighbors) != 0 or self.text is not None):
            return False
        try:
            response = session.get(WIKIPEDIA_PREFIX + self.link, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            neighbors_set = set()
            neighbor_link_list = []
            for a in soup.find_all('a', href=True):
                if self.valid_link(a['href']) and a['href'][6:] not in neighbors_set:
                    neighbor_link_list.append(a['href'][6:])
                    neighbors_set.add(a['href'][6:])
            self.neighbors = neighbor_link_list[:MAX_LINKS]

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
