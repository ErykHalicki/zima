import json
from torch import tensor

class Tokenizer:
    def __init__(self):
        self.vocabulary = {} # {token: index}
        self.inverse_vocabulary = {} # {index: token}

    def calculate_vocabulary_from_text(self, text):
        self.vocabulary.clear()
        vocabulary_set = set([char for char in text])
        vocabulary_list = list(vocabulary_set)
        vocabulary_list.sort(key=ord) # sort by unicode value to make more reproducible
        for index, key in enumerate(vocabulary_list):
            self.vocabulary[key] = index
            self.inverse_vocabulary[index] = key
        self.vocabulary["unknown"] = len(self.vocabulary)
        self.inverse_vocabulary[len(self.vocabulary)-1] = "unknown"
    
    def vocabulary_length(self):
        return len(self.vocabulary)

    def tokenize(self, text):
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
                result.append(self.vocabulary["unknown"])
        return tensor(result)

    def load_vocabulary_from_json(self, file_path):
        with open(file_path, "rt") as f:
            self.vocabulary = json.load(f)

    def save_vocabulary_to_json(self, file_path):
        with open(file_path, "wt") as f:
            json.dump(self.vocabulary, f)
        
