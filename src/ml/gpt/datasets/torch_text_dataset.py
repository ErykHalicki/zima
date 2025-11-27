from .text_dataset import TextDataset
from torch.utils.data import Dataset
from .tokenizer import PAD_TOKEN_ID, END_TOKEN_ID 
import torch

class TorchTextDataset(TextDataset, Dataset):
    def __init__(self, file_path, chunk_size=1024): 
        super().__init__(file_path)
        self.document_name_list = self.get_document_name_list()
        self.document_count = len(self.document_name_list)
        self.documents = []
        self.token_count = 0
        for doc in self.document_name_list:
            self.documents.append(self.get_document(doc))
            self.token_count+=self.documents[-1].shape[0]

        self.chunk_size = chunk_size
        self.chunks = []
        self.masks = []

        for doc in self.documents:
            num_chunks = (len(doc) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(doc))
                chunk = torch.tensor(doc[start_idx:end_idx])

                mask = torch.ones(chunk_size, dtype=torch.long)

                if len(chunk) < chunk_size:
                    padding_length = chunk_size - len(chunk)
                    chunk = torch.cat([chunk, torch.full((padding_length,), PAD_TOKEN_ID, dtype=torch.long)])
                    mask[chunk_size - padding_length:] = 0

                self.chunks.append(chunk)
                self.masks.append(mask)

        total_tokens = sum(len(chunk) for chunk in self.chunks)
        end_count = sum((chunk == END_TOKEN_ID).sum().item() for chunk in self.chunks)
        print(f"{END_TOKEN_ID} tokens: {end_count}/{total_tokens} = {100*end_count/total_tokens:.1f}%")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {'chunks': self.chunks[idx].detach(), 'masks': self.masks[idx].detach()}
        #if token at index
        
