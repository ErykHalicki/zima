import torch
import torch.nn as nn
import numpy as np 

class AttentionHead(nn.Module):
    def __init__(self,
                 d_model,
                 d_keys):
        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys
        self.root_d_keys = d_keys**(1/2)
        self.query_weights = nn.Linear(d_model, d_keys)
        self.key_weights = nn.Linear(d_model, d_keys)
        self.value_weights = nn.Linear(d_model, d_keys)
        self.softmax = nn.Softmax(dim=2) # softmax over the columns (s.t. every row sums to 1)

    def forward(self, x, mask=None):
        '''
        x: batch of stacked sequence row vectors (batch, d_seq, d_model)
        mask (Optional): if provided, will mask the attention matrix (batch, d_seq, d_seq) 1 to keep, 0 to mask
        '''
        Q = self.query_weights(x)
        K = self.key_weights(x)
        V = self.value_weights(x)
        return self.scaled_dot_product_attention(Q, K, V, mask)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        '''
        Q: Query matrix (each row represents "what is this token looking for"), (batch x d_seq x d_keys)
        K: Key matrix (each row represents "what does this token have"), (batch x d_seq x d_keys)
        V: Value matrix (each row represents "what is the information in this token?"), (batch x d_seq x d_keys)
        '''
        K_t = K.transpose(1,2) # (batch, d_seq, d_keys) -> (batch, d_keys, d_seq)
        scaled_attention_matrix = Q@K_t/self.root_d_keys # (d_seq, d_keys) x (d_keys, d_seq) -> (d_seq, d_seq)
        if mask is not None:
            scaled_attention_matrix = scaled_attention_matrix.masked_fill(mask == 0, float('-inf'))
        # attention matrix[i,j] says, "how much does token i care about token j"
        normalized_attention_matrix = self.softmax(scaled_attention_matrix) 
        # every row now represents a weight vector for each token in the sequence, between 0-1, summing to 1.  
        return normalized_attention_matrix @ V # (d_seq, d_seq) x (d_seq, d_keys) -> (d_seq, d_keys)
        # every row still corresponds to one token, but now the token has been enriched with meaning from the other tokens in the sequence that matter to it    

