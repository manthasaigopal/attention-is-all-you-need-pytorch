from utils import InputEmbeddings, PositionalEncoding, FeedForwardLayer, EncoderLayer
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, dropout, max_seq_len):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)]        
        )

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x 