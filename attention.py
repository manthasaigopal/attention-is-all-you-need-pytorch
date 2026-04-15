import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()

        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def compute_attention(self, query, key, value, mask=None):
        # query -> (b, num_heads, seq_len, head_dim)
        scores = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)

    def combine_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(x.size(0), -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.query_linear(query))
        K = self.split_heads(self.key_linear(key))
        V = self.split_heads(self.value_linear(value))
        attention_weights = self.compute_attention(Q, K, V, mask)
        output = self.combine_heads(attention_weights)
        return self.output_linear(output)


mha = MultiHeadAttention(num_heads=8, d_model=512)

# I love playing piano, I enjoy watching badminton
input = torch.randn([2, 4, 512])
out = mha(input, input, input)
print(out.shape)