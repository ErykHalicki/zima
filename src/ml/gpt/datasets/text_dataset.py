import h5py
import numpy as np
import os

class TextDataset:
    def __init__(self, file_path):
        self.file_path = os.path.expanduser(file_path)
        self.file_path = os.path.abspath(self.file_path)

        if not os.path.exists(self.file_path):
            with h5py.File(self.file_path, 'w') as f:
                pass

    def get_vocabulary(self):
        with h5py.File(self.file_path, 'r') as f:
            if "metadata" in f and "vocabulary" in f["metadata"]:
                return f["metadata"]["vocabulary"][:]
            return None

    def get_document_name_list(self):
        with h5py.File(self.file_path, 'r') as f:
            if "documents" in f:
                return list(f["documents"].keys())
            return [] 

    def add_vocabulary(self, vocabulary):
        '''
        vocabulary: numpy array of shape [vocab_size, 2] 
        where vocabulary[i, 0] = unicode character and vocabulary[i, 1] = index
        '''
        with h5py.File(self.file_path, 'a') as f:
            if "metadata" not in f:
                f.create_group('metadata')
            if "vocabulary" in f["metadata"]:
                raise Exception("Vocabulary already exists, cannot override")
            f["metadata"].create_dataset("vocabulary", data=vocabulary, compression=None)

    def get_document(self, document_name):
        with h5py.File(self.file_path, 'r') as f:
            if "documents" in f and document_name in f["documents"]:
                return f["documents"][document_name][:]
            return None

    def add_documents(self, document_dict):
        '''
        document_dict: dictionary of numpy arrays, where key is document name and data is 1D numpy array of tokenized data
        '''
        with h5py.File(self.file_path, 'a') as f:
            if "metadata" not in f or "vocabulary" not in f["metadata"]:
                raise Exception("Must add vocabulary before adding documents")
            if "documents" not in f:
                f.create_group('documents')
            for key in document_dict:
                f['documents'].create_dataset(key, data=document_dict[key], compression=None)

