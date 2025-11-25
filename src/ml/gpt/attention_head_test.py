from models.transformer import AttentionHead
import torch
d_model = 100
d_keys = d_model
seq_len = 1000
batch_size = 32
x = torch.randn(batch_size,seq_len,d_model)

a_head = AttentionHead(d_model, d_keys)
mask = torch.ones(batch_size, seq_len, seq_len).tril()

print(a_head(x, mask).shape)
