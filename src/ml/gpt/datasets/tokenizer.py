import numpy as np

UNKOWN_TOKEN= "{UNK}"
UNKOWN_TOKEN_ID = 2
END_TOKEN = "{END}"
END_TOKEN_ID = 1
PAD_TOKEN = "{PAD}"
PAD_TOKEN_ID = 0

class Tokenizer:
    def __init__(self):
        self.vocabulary = {} # {token: index}
        self.inverse_vocabulary = {} # {index: token}

    def calculate_vocabulary_from_text(self, text):
        self.vocabulary.clear()
        vocabulary_set = set([char for char in text])
        vocabulary_list = list(vocabulary_set)
        vocabulary_list.sort(key=ord) # sort by unicode value to make more reproducible
        
        self.vocabulary[PAD_TOKEN] = PAD_TOKEN_ID
        self.inverse_vocabulary[PAD_TOKEN_ID] = PAD_TOKEN
        self.vocabulary[UNKOWN_TOKEN] = UNKOWN_TOKEN_ID
        self.inverse_vocabulary[UNKOWN_TOKEN_ID] = UNKOWN_TOKEN
        self.vocabulary[END_TOKEN] = END_TOKEN_ID
        self.inverse_vocabulary[END_TOKEN_ID] = END_TOKEN

        start_index = len(self.vocabulary)
        for index, key in enumerate(vocabulary_list):
            self.vocabulary[key] = index+start_index
            self.inverse_vocabulary[index+start_index] = key
    
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
                result.append(self.vocabulary[UNKOWN_TOKEN])
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
        self.vocabulary[UNKOWN_TOKEN] = UNKOWN_TOKEN_ID
        self.inverse_vocabulary[UNKOWN_TOKEN_ID] = UNKOWN_TOKEN
        self.vocabulary[END_TOKEN] = END_TOKEN_ID
        self.inverse_vocabulary[END_TOKEN_ID] = END_TOKEN
