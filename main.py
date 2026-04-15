import sys, os
sys.path.insert(0, os.path.dirname(__file__))
 
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
 
from encoder import Encoder
from decoder import TransformerDecoder
from utils import generate_target_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_layers=6, num_heads=8, hidden_dim=2048,
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads,
            hidden_dim=hidden_dim, dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads,
            hidden_dim=hidden_dim, dropout=dropout,
            max_seq_length=max_seq_len,
        )
 
    def make_src_mask(self, src, pad_idx):
        """Mask out padding tokens in the source (batch, 1, 1, seq_len)."""
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
 
    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.make_src_mask(src, src_pad_idx)          # padding mask
        tgt_len  = tgt.size(1)
        tgt_mask = generate_target_mask(tgt_len).to(tgt.device)  # causal mask
 
        enc_output = self.encoder(src, src_mask)
        # cross_mask reuses the source padding mask so the decoder doesn't attend to encoder padding positions
        logits = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return logits


# Hyper parameteres
VOCAB_SIZE   = 50          # small vocab for the toy task
PAD_IDX      = 0
BOS_IDX      = 1           # beginning-of-sequence token
D_MODEL      = 128       
NUM_LAYERS   = 2
NUM_HEADS    = 4
HIDDEN_DIM   = 256
DROPOUT      = 0.1
MAX_SEQ_LEN  = 32
BATCH_SIZE   = 64
NUM_EPOCHS   = 20
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(batch_size, seq_len, vocab_size, pad_idx, bos_idx, device):
    # Source: random sequences, length between seq_len//2 and seq_len
    lengths = torch.randint(seq_len // 2, seq_len, (batch_size,))
    src = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for i, l in enumerate(lengths):
        src[i, :l] = torch.randint(2, vocab_size, (l,))   # 0=PAD, 1=BOS reserved
 
    # Target (teacher-forcing input): BOS + src tokens (shifted right)
    bos_col = torch.full((batch_size, 1), bos_idx, dtype=torch.long)
    tgt_input  = torch.cat([bos_col, src[:, :-1]], dim=1)   # shifted right
    tgt_output = src                                          # what we expect out
 
    return src.to(device), tgt_input.to(device), tgt_output.to(device)


def noam_schedule(step, d_model, warmup_steps=400):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

def train():
    print(f"Training on: {DEVICE}")
 
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)
 
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
 
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_schedule(step, D_MODEL, warmup_steps=400)
    )

    criterion = nn.NLLLoss(ignore_index=PAD_IDX)
 
    STEPS_PER_EPOCH = 100  
    global_step = 0
 
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
 
        for step in range(STEPS_PER_EPOCH):
            src, tgt_input, tgt_output = get_batch(
                BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE, PAD_IDX, BOS_IDX, DEVICE
            )
 
            # Forward pass
            # logits: (batch, seq_len, vocab_size)  — softmax probabilities
            logits = model(src, tgt_input, PAD_IDX, PAD_IDX)
 
            # Reshape for loss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
            log_probs = torch.log(logits + 1e-9)
            loss = criterion(
                log_probs.view(-1, VOCAB_SIZE),
                tgt_output.view(-1),
            )
 
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
 
            total_loss += loss.item()
 
        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"Epoch {epoch:>3}/{NUM_EPOCHS}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
 
    ckpt_path = "transformer_checkpoint.pt"
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                NUM_EPOCHS,
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")

 
@torch.no_grad()
def greedy_decode(model, src, max_len, bos_idx, pad_idx, device):
    """Auto-regressively generate a sequence given a source."""
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src, pad_idx)
    enc_output = model.encoder(src, src_mask)
 
    # Start with BOS token
    tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
 
    for _ in range(max_len - 1):
        tgt_len  = tgt.size(1)
        tgt_mask = generate_target_mask(tgt_len).to(device)
        probs    = model.decoder(tgt, enc_output, tgt_mask, src_mask)
        next_tok = probs[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt      = torch.cat([tgt, next_tok], dim=1)
 
    return tgt
 
 
if __name__ == "__main__":
    train()
 
    print("\n── Inference sanity check ──")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)
    ckpt = torch.load("transformer_checkpoint.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
 
    src_example = torch.tensor([[5, 10, 15, 20, 0, 0]], dtype=torch.long)
    output = greedy_decode(model, src_example, max_len=6,
                           bos_idx=BOS_IDX, pad_idx=PAD_IDX, device=DEVICE)
    print(f"src    : {src_example.tolist()}")
    print(f"decoded: {output[:, 1:].tolist()}")