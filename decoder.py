from utils import InputEmbeddings, PositionalEncoding, DecoderLayer, generate_target_mask
import torch.nn as nn
import torch.nn.functional as F
import torch

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_length)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)
  
    def forward(self, x, y, tgt_mask, cross_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, y, tgt_mask, cross_mask)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


transformer_decoder = TransformerDecoder(vocab_size=300, d_model=512, num_layers=2, num_heads=8, hidden_dim=256, dropout=0.5, max_seq_length=100)   

input_tokens = torch.tensor([[2, 3, 4,], [4, 5, 6]])
seq_len = input_tokens.shape[1]
tgt_mask = generate_target_mask(seq_len)
output = transformer_decoder(input_tokens, tgt_mask)
print(output)
print(output.shape)