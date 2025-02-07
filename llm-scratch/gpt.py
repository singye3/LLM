# Reference : llm-from-scratch book by Sebastian Raschka
import torch
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_o = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, t, c = x.size()

        keys = self.w_k(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.w_v(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.w_q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = keys @ queries.transpose(-2, -1)

        # mask out the upper half of the dot product matrix
        mask = self.mask.bool()[:t, :t]
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        scores_scaled = attn_scores / (keys.shape[-1] ** 0.5)
        attn_weights = torch.softmax(scores_scaled, dim=-1)
        
        # Shape: (b, t, b, c)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # combine heads
        context_vec = context_vec.contiguous().view(b, t, self.d_out)
        
        return self.w_o(context_vec)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.bias

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['emb_dim']*4),
            GELU(),
            nn.Linear(cfg['emb_dim']*4, cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = MLP(cfg)
        self.ln1 = LayerNorm(cfg['emb_dim'])
        self.ln2 = LayerNorm(cfg['emb_dim'])
        self.drop_residual = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        # layer norm before attention and feedforward blocks
        x = x + self.drop_residual(self.att(self.ln1(x)))
        x = x + self.drop_residual(self.ff(self.ln2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False) 

    def forward(self, in_inx):
        _, t = in_inx.size()
        pos_emb = self.pos_emb(torch.arange(t, device=in_inx.device))
        x = self.tok_emb(in_inx) + pos_emb
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x) # shape: b, t, vocab_size



