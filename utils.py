import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import MultiHeadAttention


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim, d_model):
        super().__init__()
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, hidden_dim, dropout):
        super().__init__()
        self.multi_head_atten = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.feed_forward_layer = FeedForwardLayer(hidden_dim=hidden_dim, d_model=d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        atten_output = self.multi_head_atten(x, x, x, src_mask)
        x = self.layer_norm1(x + self.dropout(atten_output))
        ff_output = self.feed_forward_layer(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x
    

class ClassificationHead(nn.Module):
    def __init__ (self, d_model, num_classes):
        super().__init__()
        self.layer = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        logits = self.layer(x)
        return F.softmax(logits, dim=-1)
    

class RegressionHead(nn.Module):
    def __init__ (self, d_model, num_classes):
        super().__init__()
        self.layer = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        return self.layer(x)


def generate_target_mask(seq_len):
    return (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, hidden_dim, dropout):
        super().__init__()
        self.multi_head_atten = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.cross_atten = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.feed_forward_layer = FeedForwardLayer(hidden_dim=hidden_dim, d_model=d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, target_mask, cross_mask):
        atten_output = self.multi_head_atten(x, x, x, target_mask)
        x = self.layer_norm1(x + self.dropout(atten_output))
        cross_output = self.cross_atten(x, y, y, cross_mask)
        x = self.layer_norm2(x + self.dropout(cross_output))
        ff_output = self.feed_forward_layer(x)
        x = self.layer_norm3(x + self.dropout(ff_output))
        return x
    

embeddings = InputEmbeddings(30, 512)
x = torch.tensor([22, 20, 5])
embeds = embeddings(x)
embeds = embeds.unsqueeze(0)
print(embeds)

pe = PositionalEncoding(d_model=512, max_seq_len=30)
out = pe(embeds)
print(out)