import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B,T,C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class AtomLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias = False)
    def forward(self,x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.token_emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
