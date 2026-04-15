<div align="center">

# Attention Is All You Need — PyTorch Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-1706.03762-b31b1b?style=flat-square)](https://arxiv.org/abs/1706.03762)

PyTorch implementation of the Transformer from scratch — no `nn.Transformer`, no shortcuts.

</div>




## Building Blocks

### MultiHeadAttention
Scaled dot-product attention projected across `num_heads` independent heads. Supports self-attention, cross-attention, and causal (masked) attention via an optional mask argument.

```python
from attention import MultiHeadAttention

attn = MultiHeadAttention(num_heads=8, d_model=512)
out  = attn(query, key, value, mask=None)  # (B, T, 512)
```


### Encoder
Stacks `num_layers` of `EncoderLayer` (multi-head self-attention → feed-forward, each with residual + LayerNorm). Prepends learned token embeddings scaled by √d and sinusoidal positional encodings.

```python
from encoder import Encoder

encoder = Encoder(vocab_size=30000, d_model=512, num_layers=6,
                  num_heads=8, hidden_dim=2048, dropout=0.1, max_seq_len=512)

enc_out = encoder(src_tokens, src_mask)  # (B, T, 512)
```


### Decoder
Stacks `num_layers` of `DecoderLayer` (masked self-attention → cross-attention over encoder output → feed-forward). Outputs a probability distribution over the target vocabulary.

```python
from decoder import TransformerDecoder

decoder = TransformerDecoder(vocab_size=30000, d_model=512, num_layers=6,
                             num_heads=8, hidden_dim=2048, dropout=0.1,
                             max_seq_length=512)

probs = decoder(tgt_tokens, enc_out, tgt_mask, src_mask)  # (B, T, vocab_size)
```


### Heads
Drop-in output heads for plugging the encoder into downstream tasks.

```python
from utils import ClassificationHead, RegressionHead

# Encoder-only classification (e.g. sentence-level)
head   = ClassificationHead(d_model=512, num_classes=10)
logits = head(enc_out[:, 0, :])   # CLS token → (B, 10)

# Regression
head = RegressionHead(d_model=512, num_classes=1)
pred = head(enc_out[:, 0, :])     # (B, 1)
```


### Masks
```python
from utils import generate_target_mask

# Causal mask — prevents decoder from attending to future positions
tgt_mask = generate_target_mask(seq_len)  # (1, T, T) bool

# Padding mask — built inside Transformer.make_src_mask()
src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
```

## Setup

```bash
git clone https://github.com/your-username/attention-is-all-you-need-pytorch.git
cd attention-is-all-you-need-pytorch
pip install torch
```

## Training

A ready-to-use training script is included with a sequence copy task, Noam LR schedule, and greedy decode. See [`main.py`](main.py) for details.

```bash
python train.py
```

## Reference

```bibtex
@article{vaswani2017attention,
  title   = {Attention Is All You Need},
  author  = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
             Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
             Kaiser, Lukasz and Polosukhin, Illia},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2017}
}
```
