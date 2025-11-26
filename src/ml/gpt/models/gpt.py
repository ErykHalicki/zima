import torch
from torch import nn
from .attention import MultiHeadAttention

MAX_SEQUENCE_LENGTH = 10000
DROPOUT_RATE = 0.3
EXPANSION_MULTIPLIER = 4 # used by transformer feedforward block
# https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model):
        self.layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_model*EXPANSION_MULTIPLIER), 
                                         nn.GELU(approximate='tanh'),
                                         nn.Linear(d_model*EXPANSION_MULTIPLIER, d_model))

    def forward(self, x, mask=None):
        normalized_x = self.layer_norm(x)
        enriched_x = self.dropout(self.multi_head_attention(normalized_x, mask))
        residual_x = torch.add(enriched_x, x) # residual connection 1
        normalized_residual_x = self.layer_norm(residual_x)
        ffn_output = self.dropout(self.feedforward(normalized_residual_x))
        return torch.add(ffn_output, normalized_residual_x) # residual connection 2

class GPT(nn.Module):
    def __init__(self, N, num_heads, d_model, vocabulary_size):
        '''
        N: Number of sequential transformer blocks to use
        num_heads: number of attention heads to use within each multi head attention block
        d_model: size of embedding representation
        vocabulary_size: max index expected by model + 1
        '''
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.positional_encodings = self.generate_positional_encoding(MAX_SEQUENCE_LENGTH)
        
        self.token_embedding = nn.Embedding(self.vocabulary_size, self.d_model)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x, mask=None):
        '''
        x: batch of token sequences (batch, d_seq) (index not one-hot)
        mask (Optional): if provided, will mask the attention matrices in multi head attention (batch, d_seq, d_seq) 1 to keep, 0 to mask
        '''
        token_embeddings = self.token_embedding(x) # turns x from (batch, d_seq) to (batch, d_seq, d_model)
        full_embeddings = torch.add(token_embeddings, self.positional_encodings[:x.shape[1]])


    def generate_positional_encoding(self, max_seq_len):
        #pre-generate (max_seq_len, d_model) positional encodings
        position_vector = torch.reshape(torch.linspace(0, max_seq_len-1, max_seq_len), (max_seq_len, 1))
        print(position_vector)
        position_matrix = torch.broadcast_to(position_vector, (max_seq_len, self.d_model)) #pos within a row is the same, since a row is a single token embedding / data point
        dimension_vector = torch.floor_divide(torch.linspace(0, self.d_model-1, self.d_model), 2)*2 # 0,2,4,...,d_model-2
        dimension_vector = torch.div(dimension_vector, self.d_model)# now its float from 0-1
        print(dimension_vector)
        dimension_matrix = torch.broadcast_to(dimension_vector, (max_seq_len, self.d_model)) # all entries in a column are the same
        dimension_matrix = torch.pow(torch.ones(max_seq_len, self.d_model)*max_seq_len, dimension_matrix) # max_seq_len**(2i/d_model)
        print(dimension_matrix)
        positional_encodings = torch.empty(max_seq_len, self.d_model)
        positional_encodings[:, ::2] = torch.sin(torch.div(position_matrix, dimension_matrix)[:,::2])
        positional_encodings[:, 1::2] = torch.cos(torch.div(position_matrix, dimension_matrix)[:,1::2])
        return positional_encodings

