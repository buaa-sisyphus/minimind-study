import torch
import torch.nn as nn
import math
from typing import Optional

def precompute_freqs_cis(dim: int, end: int=32*1024, rope_base: int=1e6, rope_scaling: Optional[dict]=None):
    # (i, i+d//2)这两维度一对，使用同一个freqs
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))

    # 做缩放，暂时没搞懂
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
        )
        if end > orig_max:
            corr_dim = next((i for i in range(dim//2) if 2*math.pi / freqs[i] > orig_max), dim//2)
            power = torch.arange(0, dim//2, device=freqs.device).float() / max(dim//2-1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            scale = torch.where(
                torch.arange(dim//2, device=freqs.device)<corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor
            )
            freqs = freqs*scale
    
    # 生成位置
    pos = torch.arange(end, device=freqs.device)
    # 计算每一个位置每一个维度的freqs
    freqs = torch.outer(pos, freqs).float() # [end, dim//2]
    # 拼接完后，在freqs_cos的第i和i+dim//2值相同，刚好对应一对
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int=1):
    def rotate_half(x):
        # [a,b]->[-b,a]
        return torch.cat(
            # -x[..., x.shape[-1]//2:] 相当于 -x[:,:,:, x.shape[-1]//2:]
            (-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]),
            dim=-1
        )
    # 假如q.shape: [bs, seq, head_num, dim]
    # 这里cos.shape: [seq, dim]
    # cos.unsqueeze(unsqueeze_dim)->[seq, 1, dim]->广播[1, seq, 1, dim]
    # 这样相乘时可以广播
    q_embed = (q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed