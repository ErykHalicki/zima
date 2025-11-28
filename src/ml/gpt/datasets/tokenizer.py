import numpy as np
from tqdm import tqdm
import json
import os

UNKNOWN_TOKEN= "{UNK}"
UNKNOWN_TOKEN_ID = 2
END_TOKEN = "{END}"
END_TOKEN_ID = 1
PAD_TOKEN = "{PAD}"
PAD_TOKEN_ID = 0

class Tokenizer:
    def __init__(self):
        self.vocabulary = {} # {token: index}
        self.inverse_vocabulary = {} # {index: token}

    def calculate_vocabulary_from_json(self, file_path, max_vocabulary_size=None):
        with open(os.path.expanduser(file_path), 'r') as f:
            token_freq = json.load(f)
        return self.calculate_vocabulary_from_token_frequency(token_freq, max_vocabulary_size)

    def get_token_frequency(self, text):
        token_freq = {}
        for char in text:
            token_freq[char] = token_freq.get(char, 0) + 1
        return token_freq

    def calculate_vocabulary_from_unicode_dataset(self, text_dataset, max_vocabulary_size=None, json_save_path=None):
        if not text_dataset.unicode_vocabulary:
            raise Exception("Input dataset must be in unicode_vocabulary mode. This method only processes unicode datasets.")

        document_names = text_dataset.get_document_name_list()

        token_freq = {}
        for doc_name in tqdm(document_names, desc="Reading documents", unit="docs"):
            unicode_array = text_dataset.get_document(doc_name)
            text = ''.join([chr(code) for code in unicode_array])
            doc_token_freq = self.get_token_frequency(text)
            for token in doc_token_freq:
                token_freq[token] = token_freq.get(token, 0) + doc_token_freq[token] 
        if json_save_path is not None:
            with open(os.path.expanduser(json_save_path), 'w') as f:
                json.dump(token_freq , f, ensure_ascii=False)
        return self.calculate_vocabulary_from_token_frequency(token_freq)

    def calculate_vocabulary_from_token_frequency(self, token_freq, max_vocabulary_size=None):
        self.vocabulary.clear()
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

        total_token_count = sum(freq for token, freq in sorted_tokens)

        if max_vocabulary_size is not None:
            max_regular_tokens = max_vocabulary_size - 3
            if max_regular_tokens < 0:
                max_regular_tokens = 0
            vocabulary_list = [token for token, freq in sorted_tokens[:max_regular_tokens]]
            excluded_tokens = sorted_tokens[max_regular_tokens:]
            unknown_count = sum(freq for token, freq in excluded_tokens)
            unknown_fraction = unknown_count / total_token_count if total_token_count > 0 else 0.0
        else:
            vocabulary_list = [token for token, freq in sorted_tokens]
            unknown_fraction = 0.0

        vocabulary_list.sort(key=ord)

        self.vocabulary[PAD_TOKEN] = PAD_TOKEN_ID
        self.inverse_vocabulary[PAD_TOKEN_ID] = PAD_TOKEN
        self.vocabulary[UNKNOWN_TOKEN] = UNKNOWN_TOKEN_ID
        self.inverse_vocabulary[UNKNOWN_TOKEN_ID] = UNKNOWN_TOKEN
        self.vocabulary[END_TOKEN] = END_TOKEN_ID
        self.inverse_vocabulary[END_TOKEN_ID] = END_TOKEN

        start_index = len(self.vocabulary)
        for index, key in enumerate(vocabulary_list):
            self.vocabulary[key] = index+start_index
            self.inverse_vocabulary[index+start_index] = key

        return unknown_fraction

    def calculate_vocabulary_from_text(self, text, max_vocabulary_size=None):
        token_freq = self.get_token_frequency(text)
        return self.calculate_vocabulary_from_token_frequency(token_freq, max_vocabulary_size)
            
    def vocabulary_length(self):
        return len(self.vocabulary)

    def tokenize(self, text, with_end_token=True):
        '''
        text: python string to be converted to list of vocabulary indices
        '''
        if len(self.vocabulary) <= 1:
            raise Exception("Must load vocabulary before tokenizing")
        result = []
        for char in text:
            if char in self.vocabulary:
                result.append(self.vocabulary[char])
            else:
                result.append(self.vocabulary[UNKNOWN_TOKEN])
        if with_end_token:
            result.append(self.vocabulary[END_TOKEN])
        return np.array(result)

    def untokenize(self, data):
        '''
        data: 1D numpy array containing tokens 
        returns: detokenized python string
        '''
        result = ""
        for token in data:
            result+=self.inverse_vocabulary[token.item()]
        return result
            
    def vocabulary_to_numpy(self):
        return np.array([[ord(token), index] for token, index in self.vocabulary.items() if token not in [UNKOWN_TOKEN, END_TOKEN, PAD_TOKEN]])
    
    def vocabulary_from_numpy(self, vocab_array):
        self.vocabulary.clear()
        self.inverse_vocabulary.clear()
        for row in vocab_array:
            token = chr(int(row[0]))
            index = int(row[1])
            self.vocabulary[token] = index
            self.inverse_vocabulary[index] = token
        self.vocabulary[PAD_TOKEN] = PAD_TOKEN_ID
        self.inverse_vocabulary[PAD_TOKEN_ID] = PAD_TOKEN
        self.vocabulary[UNKNOWN_TOKEN] = UNKNOWN_TOKEN_ID
        self.inverse_vocabulary[UNKNOWN_TOKEN_ID] = UNKNOWN_TOKEN
        self.vocabulary[END_TOKEN] = END_TOKEN_ID
        self.inverse_vocabulary[END_TOKEN_ID] = END_TOKEN
