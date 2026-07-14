import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import (
    VOCAB_SIZE, D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, ROPE_THETA, TIE_EMBEDDINGS, HEAD_DIM, KV_DIM, N_QUERIES_PER_KV, PAD_TOKEN_ID
)

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps= 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim = True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).type_as(x)

# RoPE
def precompute_freqs_cis(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles) # complex number

def apply_rotary_emb(q,k, freqs_cis):
    # q, k: (B, T, n_heads, head_dim)
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    q_out = torch.view_as_real(q_c * freqs).flatten(3).type_as(q)
    k_out = torch.view_as_real(k_c * freqs).flatten(3).type_as(k)
    return q_out, k_out

# SwiGLU
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, FFN_DIM, bias = False)
        self.up = nn.Linear(D_MODEL, FFN_DIM, bias = False)
        self.down = nn.Linear(FFN_DIM, D_MODEL, bias = False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

# Grouped QUery Attention (GQA)
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias = False)
        self.k_proj = nn.Linear(D_MODEL, N_KV_HEADS * HEAD_DIM, bias = False)
        self.v_proj = nn.Linear(D_MODEL, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM)
        K = self.k_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM)
        V = self.v_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM)

        Q, K = apply_rotary_emb(Q, K, freqs_cis) # only applies to Q, K, not V

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        K = K.repeat_interleave(N_QUERIES_PER_KV, dim = 1)
        V = V.repeat_interleave(N_QUERIES_PER_KV, dim = 1)

        # FlashAttention
        out = F.scaled_dot_product_attention(Q, K, V, is_causal = True)

        out = out.transpose(1,2).contiguous().view(B, T, D_MODEL)
        return self.o_proj(out)

# Transformer
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = FeedForward()
        self.norm_attn = RMSNorm(D_MODEL)
        self.norm_ff = RMSNorm(D_MODEL)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm_attn(x), freqs_cis)
        x = x + self.ff(self.norm_ff(x))
        return x

# AtomLM
class AtomLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias = False)

        if TIE_EMBEDDINGS:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN, ROPE_THETA)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean = 0.0, std=0.02)

            # residual projections scaling
            scale = 1.0 / math.sqrt(2 * N_LAYERS)
            for name, param in self.named_parameters():
                if 'o_proj.weight' in name or 'down.weight' in name:
                    param.data.mul_(scale)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        freqs = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs)

        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                targets.view(-1),
                ignore_index = PAD_TOKEN_ID,
            )

            return logits, loss

        # inference
        logits = self.lm_head(x[:, [-1], :])
        return logits, None


    # disabiling grad in generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_id=None):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_token], dim=1)
            if stop_token_id is not None and (next_token == stop_token_id).all():
                break
        return idx


    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
