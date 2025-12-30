import h5py
import numpy as np
import os

class TextDataset:
    def __init__(self, file_path, unicode_vocabulary=False):
        self.file_path = os.path.expanduser(file_path)
        self.file_path = os.path.abspath(self.file_path)

        if not os.path.exists(self.file_path):
            self.unicode_vocabulary = unicode_vocabulary
            with h5py.File(self.file_path, 'w') as f:
                metadata = f.create_group('metadata')
                metadata.create_dataset('unicode_vocabulary', data=np.array([1 if unicode_vocabulary else 0], dtype=np.int32))
        else:
            with h5py.File(self.file_path, 'r') as f:
                if 'metadata' in f and 'unicode_vocabulary' in f['metadata']:
                    self.unicode_vocabulary = bool(f['metadata']['unicode_vocabulary'][0])
                else:
                    self.unicode_vocabulary = False

    def get_vocabulary(self):
        with h5py.File(self.file_path, 'r') as f:
            if "metadata" in f and "vocabulary" in f["metadata"]:
                return f["metadata"]["vocabulary"][:]
            return None

    def get_unicode_vocabulary_mode(self):
        with h5py.File(self.file_path, 'r') as f:
            if "metadata" in f and "unicode_vocabulary" in f["metadata"]:
                return bool(f["metadata"]["unicode_vocabulary"][0])
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
        if self.unicode_vocabulary:
            raise Exception("Cannot add vocabulary when unicode_vocabulary mode is enabled")
        with h5py.File(self.file_path, 'a') as f:
            if "metadata" not in f:
                f.create_group('metadata')
            if "vocabulary" in f["metadata"]:
                raise Exception("Vocabulary already exists, cannot override")
            f["metadata"].create_dataset("vocabulary", data=vocabulary, compression='gzip')

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
            if not self.unicode_vocabulary:
                if "metadata" not in f or "vocabulary" not in f["metadata"]:
                    raise Exception("Must add vocabulary before adding documents")
            if "documents" not in f:
                f.create_group('documents')
            for key in document_dict:
                sanitized_key = key.replace('/', '_')
                f['documents'].create_dataset(sanitized_key, data=document_dict[key], compression='gzip')

    def add_document(self, document_name, document_data):
        '''
        document_name: string name for the document
        document_data: 1D numpy array of tokenized data (or ord() values in unicode mode)
        '''
        with h5py.File(self.file_path, 'a') as f:
            if not self.unicode_vocabulary:
                if "metadata" not in f or "vocabulary" not in f["metadata"]:
                    raise Exception("Must add vocabulary before adding documents")
            if "documents" not in f:
                f.create_group('documents')
            sanitized_name = document_name.replace('/', '_')
            if sanitized_name in f['documents']:
                return
            f['documents'].create_dataset(sanitized_name, data=document_data, compression='gzip')

    def repair_dataset(self):
        '''
        Repairs datasets where document names contain '/' characters which HDF5 interprets as group hierarchies.
        This function finds all nested groups/datasets and moves them to the top-level 'documents' group
        with '/' replaced by '_' in their names.
        '''
        def find_all_datasets(group, prefix=''):
            datasets = {}
            for key in group.keys():
                item = group[key]
                current_path = f"{prefix}/{key}" if prefix else key
                if isinstance(item, h5py.Dataset):
                    datasets[current_path] = item[:]
                elif isinstance(item, h5py.Group):
                    datasets.update(find_all_datasets(item, current_path))
            return datasets

        with h5py.File(self.file_path, 'a') as f:
            if "documents" not in f:
                return

            all_datasets = find_all_datasets(f["documents"])

            keys_to_fix = [key for key in all_datasets.keys() if '/' in key]

            if not keys_to_fix:
                return

            temp_group_name = "_temp_repair"
            if temp_group_name in f:
                del f[temp_group_name]
            temp_group = f.create_group(temp_group_name)

            for old_key in keys_to_fix:
                new_key = old_key.replace('/', '_')
                temp_group.create_dataset(new_key, data=all_datasets[old_key], compression='gzip')

            def delete_group_recursive(group, path_parts):
                if len(path_parts) == 1:
                    if path_parts[0] in group:
                        del group[path_parts[0]]
                else:
                    if path_parts[0] in group and isinstance(group[path_parts[0]], h5py.Group):
                        delete_group_recursive(group[path_parts[0]], path_parts[1:])
                        if len(group[path_parts[0]].keys()) == 0:
                            del group[path_parts[0]]

            for old_key in keys_to_fix:
                path_parts = old_key.split('/')
                delete_group_recursive(f["documents"], path_parts)

            for key in temp_group.keys():
                f["documents"].create_dataset(key, data=temp_group[key][:], compression='gzip')

            del f[temp_group_name]

