import torch
from torch import nn
from .attention import MultiHeadAttention

MAX_SEQUENCE_LENGTH = 10000
DROPOUT_RATE = 0.3
EXPANSION_MULTIPLIER = 4 # used by transformer feedforward block
# https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.feedforward = nn.Sequential(nn.Linear(d_model, d_model*EXPANSION_MULTIPLIER), 
                                         nn.GELU(approximate='tanh'),
                                         nn.Linear(d_model*EXPANSION_MULTIPLIER, d_model))

    def forward(self, batch_and_mask):
        '''
        batch_and_mask: tuple of (batch, mask)
        '''
        x, mask = batch_and_mask
        normalized_x = self.layer_norm(x)
        enriched_x = self.dropout(self.multi_head_attention(normalized_x, mask))
        residual_x = torch.add(enriched_x, x) # residual connection 1
        normalized_residual_x = self.layer_norm(residual_x)
        ffn_output = self.dropout(self.feedforward(normalized_residual_x))
        return torch.add(ffn_output, normalized_residual_x), mask # residual connection 2

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
        self.positional_encodings = nn.Parameter(self.generate_positional_encoding(MAX_SEQUENCE_LENGTH), requires_grad=False) 
        # (max_seq_lenth, d_model), needs to be reduced to (d_seq, d_model) during forward pass
        
        self.token_embedding_matrix = torch.empty(self.vocabulary_size, self.d_model)
        self.token_embedding_matrix = nn.init.xavier_uniform_(self.token_embedding_matrix, gain=nn.init.calculate_gain("relu"))
        self.token_embedding_matrix = nn.Parameter(self.token_embedding_matrix, requires_grad=True) # (batch, d_seq, d_vocab) -> (batch, d_seq, d_model)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.transformer_blocks = nn.Sequential()
        for _ in range(N):
            self.transformer_blocks.append(TransformerBlock(num_heads, self.d_model))
        self.output_projection = nn.Linear(d_model, vocabulary_size) 
        self.softmax = nn.Softmax(dim=0) #used with sequence already reduced to 1D Tensor (d_vocab)

    def forward(self, x, mask=None):
        '''
        x: batch of token sequences (batch, d_seq) (index not one-hot)
        mask (Optional): if provided, will mask the attention matrices in multi head attention (batch, d_seq, d_seq) 1 to keep, 0 to mask
        returns logits, not probabilities
        '''
        x = nn.functional.one_hot(x, num_classes=self.vocabulary_size).float() #(batch, d_seq) -> (batch, d_seq, d_vocab)
        token_embeddings = x@self.token_embedding_matrix # turns x from (batch, d_seq, d_vocab) to (batch, d_seq, d_model)
        reduced_positional_encodings = self.positional_encodings[:x.shape[1]] # reduces positional encoding matrix to (d_seq, d_model)
        full_embeddings = torch.add(token_embeddings, reduced_positional_encodings)
        full_embeddings = self.dropout(full_embeddings)
        transformer_block_chain_output, _ = self.transformer_blocks((full_embeddings, mask)) #(batch, d_seq, d_model)
        return transformer_block_chain_output@torch.transpose(self.token_embedding_matrix, 0,1) 
        # (batch, d_seq, d_model) -> (batch, d_seq, d_vocab) 
        # returns dot product of the output of the transformer block with each tokens embedded vector
        # essentially returning, "how similar is my output to each token in my vocabulary"
        # and then we softmax such that the most similar token embedding has the highest probability of being chosen

    def inference(self, x, temperature=1.0):
        '''
        x: token sequence (d_seq) (index not one-hot)
        temperature (default=1.0): higher temperature results in more even probability distribution / reduced model output confidence
        returns vocabulary index of predicted token
        '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if len(x.shape) != 1:
            raise Exception("Inference function is only meant for a single input sequence!")
        x = torch.unsqueeze(x, dim=0)
        mask = torch.unsqueeze(torch.tril(torch.ones(x.shape[0],x.shape[0])), dim=0) # (batch, d_seq, d_seq)
        logits = torch.squeeze(self.forward(x,mask))[-1]/temperature
        # remove all dimensions and entries other than the one corresponding to the last token (d_vocab)
        probabilities = self.softmax(logits) # (d_vocab) but summing to 1
        return torch.multinomial(probabilities, 1).cpu().item() 
        # sample the next token probabilities corresponding the last token in the sequence

    def generate_positional_encoding(self, max_seq_len):
        #pre-generate (max_seq_len, d_model) positional encodings
        position_vector = torch.reshape(torch.linspace(0, max_seq_len-1, max_seq_len), (max_seq_len, 1))
        position_matrix = torch.broadcast_to(position_vector, (max_seq_len, self.d_model)) #pos within a row is the same, since a row is a single token embedding / data point
        dimension_vector = torch.floor_divide(torch.linspace(0, self.d_model-1, self.d_model), 2)*2 # 0,2,4,...,d_model-2
        dimension_vector = torch.div(dimension_vector, self.d_model)# now its float from 0-1
        dimension_matrix = torch.broadcast_to(dimension_vector, (max_seq_len, self.d_model)) # all entries in a column are the same
        dimension_matrix = torch.pow(torch.ones(max_seq_len, self.d_model)*max_seq_len, dimension_matrix) # max_seq_len**(2i/d_model)
        positional_encodings = torch.empty(max_seq_len, self.d_model)
        positional_encodings[:, ::2] = torch.sin(torch.div(position_matrix, dimension_matrix)[:,::2])
        positional_encodings[:, 1::2] = torch.cos(torch.div(position_matrix, dimension_matrix)[:,1::2])
        return positional_encodings

