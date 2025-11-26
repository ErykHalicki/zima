import torch
from torch import nn
from .attention import MultiHeadAttention

MAX_SEQUENCE_LENGTH = 10000
DROPOUT_RATE = 0.3
EXPANSION_MULTIPLIER = 4
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
    def __init__(self, N, num_heads, d_model):
        '''
        N: Number of sequential transformer blocks to use
        num_heads: number of attention heads to use within each multi head attention block
        d_model: size of embedding representation
        '''
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.d_model = d_model
        self.positional_encodings = self.generate_positional_encoding(MAX_SEQUENCE_LENGTH)

    def forward(self, x, mask=None):
        '''
        x: batch of token sequences (batch, d_seq)
        '''
        # embed the inputs
        # add positional encoding to the input 
        embeddings = torch.empty(x.shape[0], x.shape[1], self.d_model)
        positional_encodings = self.positional_encodings[:x.shape[1]]

    def generate_positional_encoding(self, max_seq_len):
        #pre-generate (max_seq_len, d_model) positional encodings
        pass



