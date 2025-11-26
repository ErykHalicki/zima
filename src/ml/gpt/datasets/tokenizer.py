import json

class Tokenizer:
    def __init__(self):
        self.vocabulary = {} # {token: index}

    def calculate_vocabulary_from_text(self, text):
        self.vocabulary.clear()
        vocabulary_set = set([char for char in text])
        vocabulary_list = list(vocabulary_set)
        vocabulary_list.sort(key=ord) # sort by unicode value to make more reproducible
        for index, key in enumerate(vocabulary_list):
            self.vocabulary[key] = index
        self.vocabulary["unknown"] = len(self.vocabulary)

    def tokenize(self, text):
        '''
        text: python string to be converted to list of vocabulary indices
        '''
        if len(self.vocabulary) <= 1:
            raise Exception("Must load vocabulary before tokenizing")
        result = []
        for char in text:
            result.append(self.vocabulary[char])
        return result

    def load_vocabulary_from_json(self, file_path):
        with open(file_path, "rt") as f:
            self.vocabulary = json.load(f)

    def save_vocabulary_to_json(self, file_path):
        with open(file_path, "wt") as f:
            json.dump(self.vocabulary, f)
        
